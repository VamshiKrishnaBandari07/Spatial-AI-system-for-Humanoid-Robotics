"""Diagnose why VGGT point maps fail the native self-reprojection gate.

The gate in ``semantics.localized_surface_points`` projects every predicted world
point through the *camera head's* pose and intrinsics and compares the result with
the pixel that generated it. VGGT regresses points and cameras with separate heads,
so this measures head agreement, not metric accuracy. This module reproduces that
exact residual per view and decomposes it so the earliest cause can be identified
rather than guessed:

* a constant shift      -> principal point / coordinate-convention error
* a radial scale term   -> focal-length disagreement between the heads
* neither (large spread)-> unstructured head disagreement (e.g. reflective or
                           textureless surfaces, low confidence regions)

It never changes a threshold and reads saved runs read-only. Residuals use the same
index-coordinate convention as the gate and upstream VGGT unprojection.
Usage:  python -m spatialmemory.head_consistency --run RUN_DIR --output report.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .pixel_mapping import native_to_geometry_matrix


def _quantiles(values: np.ndarray) -> dict:
    if not len(values):
        return {"median": None, "p90": None, "p99": None}
    q = np.quantile(values, [0.5, 0.9, 0.99])
    return {"median": float(q[0]), "p90": float(q[1]), "p99": float(q[2])}


def head_consistency(
    world_points: np.ndarray,
    confidence: np.ndarray,
    extrinsic: np.ndarray,
    intrinsic: np.ndarray,
    mapping: np.ndarray,
    valid_pixels: np.ndarray | None = None,
    limit_native_px: float = 2.0,
) -> dict:
    """Residual of one view: HxWx3 points, HxW confidence, 3x4 T_camera_world, 3x3 K.

    ``mapping`` is the native->model-input affine from ``native_to_geometry_matrix``.
    Pixels are restricted to ``valid_pixels`` and to confidence at or above the
    median of those pixels, mirroring the production gate. Nothing is calibrated.
    """
    height, width = world_points.shape[:2]
    content = np.ones((height, width), bool) if valid_pixels is None else valid_pixels.astype(bool)
    yy, xx = np.indices((height, width))
    camera = world_points @ extrinsic[:, :3].T + extrinsic[:, 3]
    projected = camera @ intrinsic.T
    with np.errstate(divide="ignore", invalid="ignore"):
        uv = projected[..., :2] / projected[..., 2, None]
    residual = uv - np.stack([xx, yy], axis=-1)  # model-input pixels
    finite = np.isfinite(residual).all(-1) & (camera[..., 2] > 0)
    selected = content & finite
    if not selected.any():
        raise ValueError("No finite positive-depth pixels in the content region")
    selected &= confidence >= np.quantile(confidence[content & finite], 0.5)
    inverse_linear = np.linalg.inv(mapping)[:2, :2]
    native_error = np.linalg.norm(residual[selected] @ inverse_linear.T, axis=-1)
    model_error = np.linalg.norm(residual[selected], axis=-1)

    fx, fy, cx, cy = intrinsic[0, 0], intrinsic[1, 1], intrinsic[0, 2], intrinsic[1, 2]
    xs, ys, res = xx[selected] - cx, yy[selected] - cy, residual[selected]
    keep = model_error <= np.quantile(model_error, 0.95)  # trim outliers before fitting
    design_u = np.stack([xs[keep], np.ones(keep.sum())], axis=1)
    design_v = np.stack([ys[keep], np.ones(keep.sum())], axis=1)
    (sx, tx), *_ = np.linalg.lstsq(design_u, res[keep, 0], rcond=None)
    (sy, ty), *_ = np.linalg.lstsq(design_v, res[keep, 1], rcond=None)
    explained = 1.0 - float(
        (np.var(res[keep, 0] - design_u @ [sx, tx]) + np.var(res[keep, 1] - design_v @ [sy, ty]))
        / max(np.var(res[keep, 0]) + np.var(res[keep, 1]), 1e-12)
    )
    return {
        "pixels_evaluated": int(selected.sum()),
        "residual_model_px": _quantiles(model_error),
        "residual_native_px": _quantiles(native_error),
        "fraction_within_native_limit": float(np.mean(native_error < limit_native_px)),
        "native_limit_px": limit_native_px,
        "native_limit_in_model_px": float(limit_native_px / np.linalg.svd(inverse_linear, compute_uv=False).max()),
        "model_limit_definition": "conservative inscribed radius; classification uses full native residual",
        "fit": {
            "relative_focal_error_x": float(sx),
            "relative_focal_error_y": float(sy),
            "shift_x_model_px": float(tx),
            "shift_y_model_px": float(ty),
            "fraction_of_variance_explained_by_shift_and_focal": explained,
        },
        "intrinsics": {"fx": float(fx), "fy": float(fy), "cx": float(cx), "cy": float(cy)},
        "mean_residual_vector_model_px": [float(res[:, 0].mean()), float(res[:, 1].mean())],
    }


def interpret(view: dict) -> str:
    """One-line reading of a view's decomposition; a hint, not a verdict."""
    fit, within = view["fit"], view["fraction_within_native_limit"]
    if within >= 0.9:
        return "passes the gate"
    shift = max(abs(fit["shift_x_model_px"]), abs(fit["shift_y_model_px"]))
    focal = max(abs(fit["relative_focal_error_x"]), abs(fit["relative_focal_error_y"]))
    explained = fit["fraction_of_variance_explained_by_shift_and_focal"]
    if explained < 0.3:
        return "unstructured head disagreement (not a shift or focal error)"
    if shift > 1.0 and focal * 259 < shift:
        return "dominated by a constant shift: check principal point / coordinate convention"
    if focal * 259 >= shift:
        return "dominated by radial scale: check focal length agreement between heads"
    return "mixed shift and focal error"


def analyse_run(run: Path) -> dict:
    """Run on a saved, completed VGGT run directory (read-only)."""
    import cv2
    from .contracts import Frame
    from .ingestion import neural_keyframes

    run = run.resolve()
    report = json.loads((run / "report.json").read_text())
    config = report["config"]
    limit = float(config.get("triangulation_max_reprojection_px", 2.0))
    result: dict = {"run": str(run), "preprocessing": config["neural_preprocessing"], "views": []}
    for role in ("before", "after"):
        manifest = json.loads((run / role / "manifest.json").read_text())
        frames = neural_keyframes(
            [Frame(**f) for f in manifest["frames"]], config["neural_frames_per_video"]
        )
        with np.load(run / "neural" / role / "predictions.npz", allow_pickle=False) as arrays:
            for i, frame in enumerate(frames):
                image = cv2.imread(str(Path(frame.path)))
                if image is None:
                    raise ValueError(f"Missing source frame {frame.id}")
                shape = arrays["world_points"].shape[2:4]
                mapping = native_to_geometry_matrix(image.shape[:2], shape, config["neural_preprocessing"])
                native = np.stack([*np.indices(shape)[::-1], np.ones(shape)], -1) @ np.linalg.inv(mapping).T
                content = (native[..., 0] >= 0) & (native[..., 0] < image.shape[1]) & (
                    native[..., 1] >= 0) & (native[..., 1] < image.shape[0])
                view = head_consistency(
                    arrays["world_points"][0, i], arrays["world_points_conf"][0, i],
                    arrays["extrinsics"][0, i], arrays["intrinsics"][0, i], mapping, content, limit,
                )
                view.update(role=role, frame_id=frame.id, native_shape=list(image.shape[:2]))
                view["reading"] = interpret(view)
                result["views"].append(view)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--output", type=Path, help="New JSON file (never overwritten)")
    args = parser.parse_args()
    result = analyse_run(args.run)
    for v in result["views"]:
        print(f"{v['role']:6} {v['frame_id']:>14}  native<{v['native_limit_px']:.0f}px: "
              f"{v['fraction_within_native_limit']:6.1%}  median {v['residual_native_px']['median']:.2f} native px"
              f" ({v['residual_model_px']['median']:.2f} model px)  -> {v['reading']}")
    if args.output:
        if args.output.exists():
            raise SystemExit(f"{args.output} exists; choose a new path")
        args.output.write_text(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
