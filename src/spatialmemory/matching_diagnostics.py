"""Auditable LoFTR pair diagnostics. Does not change production match gates.

Phase-3 style measurement: record resized shapes, raw model returns, score
histograms, and rejection counts at every wrapper filter. Raw scores are not
calibrated probabilities. The production acceptance threshold remains 0.8.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from numpy.typing import NDArray

from .learned_matching import CHECKPOINT_SHA256, MODEL_REVISION, prepare_crop


PRODUCTION_SCORE_THRESHOLD = 0.8


def score_histogram(scores: NDArray[np.floating], edges: list[float] | None = None) -> dict[str, int]:
    """Bucket raw matcher scores for reports; empty input yields empty counts."""
    bins = edges or [0.0, 0.2, 0.4, 0.6, 0.8, 1.01]
    values = np.asarray(scores, dtype=float).ravel()
    values = values[np.isfinite(values)]
    counts, _ = np.histogram(values, bins=bins) if len(values) else (np.zeros(len(bins) - 1, int), bins)
    return {
        f"[{bins[i]:.1f},{bins[i + 1]:.1f})": int(counts[i]) for i in range(len(counts))
    }


def filter_stages(
    points_a: NDArray[np.floating],
    points_b: NDArray[np.floating],
    scores: NDArray[np.floating],
    mask_a: NDArray[np.uint8],
    mask_b: NDArray[np.uint8],
    score_threshold: float = PRODUCTION_SCORE_THRESHOLD,
) -> dict[str, Any]:
    """Apply the same wrapper filters as LearnedCropMatcher without loading weights.

    Returns per-stage keep counts and native endpoints that survive all gates.
    """
    pa = np.asarray(points_a, dtype=float).reshape(-1, 2)
    pb = np.asarray(points_b, dtype=float).reshape(-1, 2)
    scores = np.asarray(scores, dtype=float).ravel()
    if not (len(pa) == len(pb) == len(scores)):
        raise ValueError("keypoints and scores must share length")
    stages = {
        "model_returned": len(scores),
        "finite_endpoints_and_scores": 0,
        "score_at_or_above_threshold": 0,
        "inside_mask_a": 0,
        "inside_mask_b": 0,
        "accepted": 0,
    }
    if not len(scores):
        return {
            "score_threshold": score_threshold,
            "score_threshold_is_calibrated_probability": False,
            "stages": stages,
            "score_histogram": score_histogram(scores),
            "accepted_native_a": [],
            "accepted_native_b": [],
            "accepted_scores": [],
        }
    finite = (
        np.isfinite(pa).all(axis=1)
        & np.isfinite(pb).all(axis=1)
        & np.isfinite(scores)
    )
    stages["finite_endpoints_and_scores"] = int(finite.sum())
    score_ok = finite & (scores >= score_threshold)
    stages["score_at_or_above_threshold"] = int(score_ok.sum())

    def inside(points: NDArray[np.floating], mask: NDArray[np.uint8]) -> NDArray[np.bool_]:
        xy = np.rint(np.nan_to_num(points)).astype(int)
        valid = (
            (xy[:, 0] >= 0)
            & (xy[:, 1] >= 0)
            & (xy[:, 0] < mask.shape[1])
            & (xy[:, 1] < mask.shape[0])
        )
        hit = np.zeros(len(xy), bool)
        hit[valid] = mask[xy[valid, 1], xy[valid, 0]] > 0
        return hit

    in_a = score_ok & inside(pa, mask_a)
    in_b = in_a & inside(pb, mask_b)
    stages["inside_mask_a"] = int(in_a.sum())
    stages["inside_mask_b"] = int(in_b.sum())
    stages["accepted"] = int(in_b.sum())
    ids = np.where(in_b)[0]
    ids = ids[np.argsort(scores[ids])[::-1]][:100]
    return {
        "score_threshold": score_threshold,
        "score_threshold_is_calibrated_probability": False,
        "stages": stages,
        "score_histogram": score_histogram(scores),
        "accepted_native_a": pa[ids].tolist(),
        "accepted_native_b": pb[ids].tolist(),
        "accepted_scores": scores[ids].tolist(),
    }


def load_gray(path: Path) -> NDArray[np.uint8]:
    image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError(f"Could not read grayscale frame: {path}")
    return image


def diagnose_pair(
    path_a: Path,
    path_b: Path,
    checkpoint: Path | None,
    score_threshold: float = PRODUCTION_SCORE_THRESHOLD,
    full_frame_masks: bool = True,
) -> dict[str, Any]:
    """Run one pair with full-frame masks (default) and original prepare_crop."""
    gray_a = load_gray(path_a)
    gray_b = load_gray(path_b)
    if full_frame_masks:
        mask_a = np.ones(gray_a.shape, np.uint8) * 255
        mask_b = np.ones(gray_b.shape, np.uint8) * 255
    else:
        raise ValueError("Object masks are not supplied in this diagnostic entrypoint")
    crop_a, affine_a = prepare_crop(gray_a, mask_a)
    crop_b, affine_b = prepare_crop(gray_b, mask_b)
    report: dict[str, Any] = {
        "frame_a": str(path_a),
        "frame_b": str(path_b),
        "native_shapes_hw": [list(gray_a.shape), list(gray_b.shape)],
        "prepared_shapes_hw": [list(crop_a.shape[-2:]), list(crop_b.shape[-2:])],
        "full_frame_masks": True,
        "imread": "IMREAD_GRAYSCALE",
        "loftr_revision": MODEL_REVISION,
        "checkpoint_sha256_expected": CHECKPOINT_SHA256,
        "production_score_threshold_unchanged": score_threshold == PRODUCTION_SCORE_THRESHOLD,
        "checkpoint_loaded": False,
        "model_returned_matches_already_internally_filtered": True,
    }
    if checkpoint is None:
        report["status"] = "prepared_only"
        report["blocker"] = "LoFTR checkpoint path required for live inference"
        report["filter_replay"] = filter_stages(
            np.zeros((0, 2)),
            np.zeros((0, 2)),
            np.zeros(0),
            mask_a,
            mask_b,
            score_threshold,
        )
        return report
    from .learned_matching import LearnedCropMatcher
    import torch

    matcher = LearnedCropMatcher(Path(checkpoint))
    report["checkpoint_loaded"] = True
    report["checkpoint_path"] = str(Path(checkpoint).resolve())
    with torch.inference_mode():
        result = matcher.model(
            {
                "image0": torch.from_numpy(crop_a),
                "image1": torch.from_numpy(crop_b),
            }
        )
    pa = result["keypoints0"].cpu().numpy()
    pb = result["keypoints1"].cpu().numpy()
    scores = result["confidence"].cpu().numpy()
    pa_native = (np.c_[pa, np.ones(len(pa))] @ affine_a.T)[:, :2]
    pb_native = (np.c_[pb, np.ones(len(pb))] @ affine_b.T)[:, :2]
    report["raw_model_match_count"] = int(len(scores))
    report["raw_score_max"] = float(scores.max()) if len(scores) else None
    report["raw_score_mean"] = float(scores.mean()) if len(scores) else None
    report["filter_replay"] = filter_stages(
        pa_native, pb_native, scores, mask_a, mask_b, score_threshold
    )
    report["status"] = "measured"
    # Production path still uses LearnedCropMatcher.match; record its accepted count.
    accepted_a, accepted_b = matcher.match(gray_a, mask_a, gray_b, mask_b)
    report["production_match_accepted"] = int(len(accepted_a))
    report["production_match_agrees_with_filter_replay"] = len(accepted_a) == report[
        "filter_replay"
    ]["stages"]["accepted"]
    return report


DEFAULT_AFTER_PAIRS = (
    (1020, 2040),
    (1020, 1440),
    (1440, 2040),
    (2040, 2640),
)


def resolve_after_frame(run_dir: Path, frame_id: int) -> Path:
    """Locate after:<id> under a completed run's after frames."""
    after = run_dir / "after"
    candidates = [
        after / f"after:{frame_id}.jpg",
        after / f"{frame_id}.jpg",
        after / f"frame_{frame_id:06d}.jpg",
    ]
    for path in candidates:
        if path.is_file():
            return path
    # Fall back to manifest lookup by id / timestamp index.
    manifest = after / "manifest.json"
    if manifest.is_file():
        frames = json.loads(manifest.read_text())["frames"]
        for row in frames:
            if row.get("id") in {f"after:{frame_id}", str(frame_id)} or row.get(
                "index"
            ) == frame_id:
                path = Path(row["path"])
                if path.is_file():
                    return path
    raise FileNotFoundError(
        f"AFTER frame {frame_id} not found under {after} "
        f"(tried {[c.name for c in candidates]})"
    )


def diagnose_after_pairs(
    run_dir: Path,
    checkpoint: Path | None,
    pairs: tuple[tuple[int, int], ...] = DEFAULT_AFTER_PAIRS,
    output: Path | None = None,
) -> dict[str, Any]:
    run_dir = Path(run_dir)
    rows = []
    for left, right in pairs:
        try:
            path_a = resolve_after_frame(run_dir, left)
            path_b = resolve_after_frame(run_dir, right)
            row = diagnose_pair(path_a, path_b, checkpoint)
            row["pair"] = [left, right]
        except FileNotFoundError as error:
            row = {
                "pair": [left, right],
                "status": "blocked",
                "blocker": str(error),
            }
        rows.append(row)
    report = {
        "method": "LoFTR full-frame grayscale diagnostic",
        "production_score_threshold": PRODUCTION_SCORE_THRESHOLD,
        "gates_unchanged": True,
        "calibrated": False,
        "run_dir": str(run_dir.resolve()),
        "pairs": rows,
    }
    if output is not None:
        output = Path(output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(report, indent=2))
    return report


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, help="Completed run with after/ frames")
    parser.add_argument("--checkpoint", type=Path, help="Verified LoFTR .ckpt path")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--pair",
        action="append",
        metavar="A-B",
        help="Optional AFTER index pair, e.g. 1020-2040 (repeatable)",
    )
    args = parser.parse_args(argv)
    if args.run is None:
        raise SystemExit(
            "Provide --run pointing at a completed inference directory with AFTER frames"
        )
    pairs = DEFAULT_AFTER_PAIRS
    if args.pair:
        parsed = []
        for item in args.pair:
            a, b = item.split("-", 1)
            parsed.append((int(a), int(b)))
        pairs = tuple(parsed)
    report = diagnose_after_pairs(args.run, args.checkpoint, pairs, args.output)
    print(json.dumps({"output": str(args.output), "pair_count": len(report["pairs"])}))


if __name__ == "__main__":
    main()
