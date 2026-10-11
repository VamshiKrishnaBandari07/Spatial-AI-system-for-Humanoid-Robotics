"""VGGT to common geometry contract; confidence is raw, not a probability."""

import sys
from pathlib import Path
import numpy as np
from .geometry import Reconstruction, extract
from .models import VGGTExpert

SOURCE_COMMIT = "a288dd0f14786c93483e45524328726ab7b1b4ce"
CHECKPOINT_REVISION = "860abec7937da0a4c03c41d3c269c366e82abdf9"


def reconstruct_neural(
    frames, output, expert=None, predictions=None, preprocessing="crop"
):
    """Bounded selected frames; preserve neural raw predictions and lookup provenance."""
    import cv2
    from vggt.utils.load_fn import load_and_preprocess_images

    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    if predictions is None:
        if expert is None:
            raise ValueError("Neural model or validated cached predictions required")
        result = expert.infer([f.path for f in frames], preprocessing=preprocessing)
    else:
        result = predictions
    arrays = {k: v for k, v in result.items() if isinstance(v, np.ndarray)}
    if not all(np.isfinite(a).all() for a in arrays.values()):
        raise ValueError("Neural geometry contains nonfinite outputs")
    np.savez_compressed(output / "predictions.npz", **arrays)
    images = (
        load_and_preprocess_images([f.path for f in frames], mode=preprocessing)
        .numpy()
        .transpose(0, 2, 3, 1)
    )
    if predictions is not None:
        reference = Path(predictions["reuse_provenance"]["processed_directory"])
        for i, image in enumerate(images):
            old = cv2.imread(str(reference / f"processed_{i:03d}.png"))
            expected_image = (image * 255).astype(np.uint8)[:, :, ::-1]
            if old is None or not np.array_equal(old, expected_image):
                raise ValueError("Cached frame content/order differs")
    expected = (1, len(frames), images.shape[1], images.shape[2], 3)
    if result["world_points"].shape != expected:
        raise ValueError("Cached point maps do not match selected frame geometry")
    # Feature coordinates must refer to the exact model input crop/resize.
    normalized = []
    for i, frame in enumerate(frames):
        path = output / f"processed_{i:03d}.png"
        cv2.imwrite(str(path), (images[i] * 255).astype(np.uint8)[:, :, ::-1])
        normalized.append(frame.model_copy(update={"path": str(path)}))
    features = extract(normalized)
    points = []
    colors = []
    tracks = {}
    poses = {}
    raw_conf = []
    content_masks = []
    for i, frame in enumerate(frames):
        xyz = result["world_points"][0, i]
        conf = result["world_points_conf"][0, i]
        content = np.ones(conf.shape, bool)
        if preprocessing == "pad":
            original = cv2.imread(frame.path)
            height, width = original.shape[:2]
            if width >= height:
                nw, nh = 518, round(height * (518 / width) / 14) * 14
            else:
                nh, nw = 518, round(width * (518 / height) / 14) * 14
            top, left = (518 - nh) // 2, (518 - nw) // 2
            content[:] = False
            content[top : top + nh, left : left + nw] = True
        content_masks.append(content)
        threshold = float(np.quantile(conf[content], 0.5))
        coords = features[frame.id][0]
        pose = np.eye(4)
        pose[:3] = result["extrinsics"][0, i]
        poses[frame.id] = pose
        for k, uv in enumerate(coords):
            x, y = np.rint(uv).astype(int)
            if (
                not (0 <= y < xyz.shape[0] and 0 <= x < xyz.shape[1])
                or not content[y, x]
                or conf[y, x] < threshold
            ):
                continue
            pid = len(points)
            points.append(xyz[y, x])
            colors.append(images[i, y, x])
            raw_conf.append(float(conf[y, x]))
            tracks[frame.id, k] = pid
    if len(points) < 20:
        raise ValueError(
            "Insufficient neural surface keypoints after raw confidence filtering"
        )
    r = Reconstruction(
        np.array(points),
        np.array(colors),
        poses,
        tracks,
        features,
        result["intrinsics"][0, 0],
        {
            "reuse_provenance": result.get("reuse_provenance"),
            "predictions_origin": "validated cached run"
            if predictions is not None
            else "pretrained inference in current run",
            "backend": "VGGT",
            "preprocessing": preprocessing,
            "padding_is_geometry_evidence": False,
            "points": len(points),
            "registered_frames": len(frames),
            "selected_frames": len(frames),
            "scale": "unverified monocular",
            "confidence_filter": "per-frame median raw confidence; not calibrated",
            "raw_confidence_range": [min(raw_conf), max(raw_conf)],
            "checkpoint_revision": CHECKPOINT_REVISION,
            "source_commit": SOURCE_COMMIT,
            "intrinsics_source": "VGGT prediction; accuracy unvalidated",
            "predictions_file": str(output / "predictions.npz"),
        },
    )
    r.surfaces = {
        frame.id: {
            "points": result["world_points"][0, i],
            "depth": result["depth"][0, i, :, :, 0],
            "confidence": result["world_points_conf"][0, i],
            "valid_pixels": content_masks[i],
            "intrinsics": result["intrinsics"][0, i],
            "depth_source": "VGGT prediction; not measured depth",
            "depth_confidence": result["depth_conf"][0, i]
            if "depth_conf" in result
            else None,
        }
        for i, frame in enumerate(frames)
    }
    return r


def load_expert():
    import os, subprocess, hashlib, torch
    from huggingface_hub import hf_hub_download, HfApi

    os.environ.setdefault("HF_HOME", "/workspace/spatialmemory-models/hf-home")
    os.environ.setdefault("HF_XET_CACHE", "/workspace/spatialmemory-models/xet")
    source = Path("/workspace/spatialmemory-thirdparty/vggt")
    actual = subprocess.check_output(
        ["git", "-C", str(source), "rev-parse", "HEAD"], text=True
    ).strip()
    if actual != SOURCE_COMMIT:
        raise ValueError("VGGT source revision mismatch; use reviewed pinned source")
    sys.path.insert(0, str(source))
    torch.set_num_threads(4)
    path = hf_hub_download(
        "facebook/VGGT-1B",
        "model.pt",
        revision=CHECKPOINT_REVISION,
        cache_dir="/workspace/spatialmemory-models",
    )
    info = HfApi().model_info(
        "facebook/VGGT-1B", revision=CHECKPOINT_REVISION, files_metadata=True
    )
    expected = next(f.lfs.sha256 for f in info.siblings if f.rfilename == "model.pt")
    digest = hashlib.sha256()
    with open(path, "rb") as file:
        for b in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(b)
    if digest.hexdigest() != expected:
        raise ValueError("VGGT checkpoint SHA-256 mismatch")
    return VGGTExpert(path)


def reusable_predictions(root, config, before_manifest, after_manifest):
    import json, subprocess
    from .ingestion import sha256

    root = Path(root).resolve()
    report = json.loads((root / "report.json").read_text())
    if (
        report["config"].get("neural_preprocessing", "crop")
        != config.neural_preprocessing
    ):
        raise ValueError("Cached geometry preprocessing differs")
    keys = [
        "geometry_backend",
        "neural_frames_per_video",
        "max_frames",
        "max_width",
        "sample_seconds",
    ]
    if any(report["config"].get(k) != getattr(config, k) for k in keys):
        raise ValueError("Cached geometry configuration differs")
    if (
        report["inputs"][0]["sha256"] != before_manifest["sha256"]
        or report["inputs"][1]["sha256"] != after_manifest["sha256"]
    ):
        raise ValueError("Cached geometry belongs to different videos")
    source = Path("/workspace/spatialmemory-thirdparty/vggt")
    if (
        subprocess.check_output(
            ["git", "-C", str(source), "rev-parse", "HEAD"], text=True
        ).strip()
        != SOURCE_COMMIT
    ):
        raise ValueError("Cached geometry source revision differs")
    sys.path.insert(0, str(source))
    result = {}
    for state in ["before", "after"]:
        diag = report["geometry"][state]
        diag = diag.get("neural_cross_validation") or diag
        if (
            diag.get("checkpoint_revision") != CHECKPOINT_REVISION
            or diag.get("source_commit") != SOURCE_COMMIT
        ):
            raise ValueError("Cached checkpoint provenance differs")
        path = root / "neural" / state / "predictions.npz"
        with np.load(path, allow_pickle=False) as arrays:
            result[state] = {k: arrays[k] for k in arrays.files}
        result[state]["reuse_provenance"] = {
            "report": str(root / "report.json"),
            "report_sha256": sha256(root / "report.json"),
            "predictions_sha256": sha256(path),
            "processed_directory": str(path.parent),
        }
    return result
