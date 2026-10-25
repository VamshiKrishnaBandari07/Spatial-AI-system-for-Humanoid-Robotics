"""Export computed dense VGGT surfaces in the accepted shared coordinate frame.

This is a visualization export, not new reconstruction or change detection.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import cv2
import numpy as np
from .contracts import Frame
from .ingestion import neural_keyframes
from .pixel_mapping import native_to_geometry_matrix
from .semantics import localized_surface_points
from .exports import ply
from .viewer import write_viewer


def transform_after(points: np.ndarray, registration: dict) -> np.ndarray:
    """Apply X_before = scale * R @ X_after + t to an (N,3) point array."""
    if not registration.get('valid'):
        raise ValueError('Shared export requires accepted registration')
    scale=float(registration['scale']);rotation=np.asarray(registration['rotation'],float)
    translation=np.asarray(registration['translation'],float)
    if (not np.isfinite(scale) or scale<=0 or rotation.shape!=(3,3)
        or translation.shape!=(3,) or not np.isfinite(rotation).all()
        or not np.isfinite(translation).all()
        or not np.allclose(rotation.T@rotation,np.eye(3),atol=1e-5)
        or not np.isclose(np.linalg.det(rotation),1,atol=1e-5)):
        raise ValueError('Invalid positive-scale Sim(3) registration')
    points=np.asarray(points,float)
    if points.ndim!=2 or points.shape[1]!=3 or not np.isfinite(points).all():
        raise ValueError('Expected finite Nx3 points')
    return scale*(points@rotation.T)+translation


def build_dense_clouds(run: Path, stride: int = 2) -> tuple[dict, dict, list, dict]:
    """Build colored dense surfaces from a completed VGGT run (visualization only)."""
    if stride < 1:
        raise ValueError("Stride must be positive")
    run = run.resolve()
    report = json.loads((run / "report.json").read_text())
    config = report["config"]
    if (
        report["status"] != "complete"
        or config["neural_frame_selection"] != "quality_diverse"
        or config["geometry_backend"] != "vggt"
        or config.get("metric_geometry_enabled", False)
    ):
        raise ValueError(
            "Completed quality-diverse VGGT run without sensor replacement required"
        )
    transform_after(np.empty((0, 3)), report["registration"])
    clouds = {}
    colors = {}
    audits = []
    for role in ["before", "after"]:
        frames = neural_keyframes(
            [
                Frame(**f)
                for f in json.loads((run / role / "manifest.json").read_text())[
                    "frames"
                ]
            ],
            config["neural_frames_per_video"],
        )
        points_list = []
        color_list = []
        with np.load(run / "neural" / role / "predictions.npz", allow_pickle=False) as arrays:
            if len(frames) != arrays["world_points"].shape[1]:
                raise ValueError("Frame count mismatch")
            for i, frame in enumerate(frames):
                path = Path(frame.path).resolve()
                if not path.is_relative_to(run):
                    raise ValueError("Frame outside source run")
                image = cv2.imread(str(path))
                processed = cv2.imread(str(run / "neural" / role / f"processed_{i:03d}.png"))
                if image is None or processed is None:
                    raise ValueError("Missing source image")
                depth = arrays["depth"][0, i, :, :, 0]
                shape = depth.shape
                mapping = native_to_geometry_matrix(
                    image.shape[:2], shape, config["neural_preprocessing"]
                )
                yy, xx = np.indices(shape)
                native = np.stack([xx, yy, np.ones_like(xx)], axis=-1) @ np.linalg.inv(
                    mapping
                ).T
                content = (
                    (native[..., 0] >= 0)
                    & (native[..., 0] < image.shape[1])
                    & (native[..., 1] >= 0)
                    & (native[..., 1] < image.shape[0])
                )
                pose = np.eye(4)
                pose[:3] = arrays["extrinsics"][0, i]
                surface = dict(
                    points=arrays["world_points"][0, i],
                    confidence=arrays["world_points_conf"][0, i],
                    depth=depth,
                    depth_confidence=arrays["depth_conf"][0, i],
                    intrinsics=arrays["intrinsics"][0, i],
                    valid_pixels=content,
                    native_to_geometry=mapping,
                    max_reprojection_px=config["triangulation_max_reprojection_px"],
                )
                mask = np.zeros(shape, bool)
                mask[::stride, ::stride] = True
                points, audit, support = localized_surface_points(
                    mask, surface, pose, config["voxel_size"], True
                )
                points_list.append(points)
                color_list.append(processed[support, ::-1] / 255.0)
                audits.append(dict(frame=frame.id, points=len(points), **audit))
        clouds[role] = np.concatenate(points_list)
        colors[role] = np.concatenate(color_list)
    clouds["after"] = transform_after(clouds["after"], report["registration"])
    if any(len(points) == 0 for points in clouds.values()):
        raise ValueError("No supported dense points in one recording")
    return clouds, colors, audits, report


def write_dense_world(run: Path, stride: int = 3) -> dict:
    """Write world_dense.npz beside world.npz for the interactive room viewer.

    Requires ``run/report.json`` already written with ``status: complete``.
    Call this only after the pipeline atomically publishes the report.
    """
    clouds, colors, audits, report = build_dense_clouds(run, stride=stride)
    path = run / "world_dense.npz"
    np.savez_compressed(
        path,
        before=clouds["before"].astype(np.float32),
        after=clouds["after"].astype(np.float32),
        before_colors=colors["before"].astype(np.float32),
        after_colors=colors["after"].astype(np.float32),
    )
    summary = {
        "path": str(path),
        "before_points": int(len(clouds["before"])),
        "after_points": int(len(clouds["after"])),
        "stride": stride,
        "frames": audits,
        "purpose": "dashboard room visualization only; not used for change decisions",
    }
    (run / "world_dense_summary.json").write_text(json.dumps(summary, indent=2))
    return summary


def export_dense(run: Path, output: Path, stride: int = 2) -> dict:
    """Filter actual point maps by existing confidence and native reprojection gates."""
    clouds, colors, audits, report = build_dense_clouds(run, stride=stride)
    if output.exists():
        raise ValueError("Use a new output directory")
    from scipy.spatial import cKDTree

    overlap = {}
    for source, target in [("before", "after"), ("after", "before")]:
        distances, _ = cKDTree(clouds[target]).query(clouds[source])
        overlap[source + "_to_" + target] = {
            "median_distance": float(np.median(distances)),
            "p95_distance": float(np.quantile(distances, 0.95)),
            "fraction_within_registration_threshold": float(
                np.mean(distances < report["registration"]["threshold"])
            ),
        }
    output.mkdir(parents=True)
    for role in clouds:
        ply(output / f"{role}.ply", clouds[role], colors[role])
    ply(
        output / "aligned_scene.ply",
        np.concatenate(list(clouds.values())),
        np.concatenate(list(colors.values())),
    )
    # Geometry-only viewer: do not copy unverified object positions onto this export.
    write_viewer(
        output,
        {
            "scene_id": report["scene_id"] + " — dense geometry; scale unverified",
            "events": [],
        },
        clouds["before"],
        clouds["after"],
    )
    viewer = output / "visualization/index.html"
    html = viewer.read_text().replace(
        "SpatialMemory-X · Scene evidence", "SpatialMemory-X · Dense geometry preview"
    )
    html = html.replace(
        "Drag to rotate · Scroll to zoom · Select a decision for its evidence",
        "Drag to rotate · Scroll to zoom · Geometry-only export; no change decisions shown",
    )
    label = (
        f"Computed geometry: {len(clouds['before']):,} before points and "
        f"{len(clouds['after']):,} after points. Scale unverified. "
        "Change decisions are not shown in this geometry-only preview."
    )
    html = html.replace(
        "</script>",
        "document.getElementById('summary').textContent="
        + json.dumps(label)
        + ";</script>",
    )
    viewer.write_text(html)
    summary = {
        "source_run": str(run),
        "source_predictions_reused": True,
        "fresh_neural_inference_in_export": False,
        "scale": "unverified reconstruction units",
        "before_points": len(clouds["before"]),
        "after_points": len(clouds["after"]),
        "registration": report["registration"],
        "dense_overlap": overlap,
        "overlap_interpretation": "Nearest-neighbor distances in unverified units; affected by partial coverage and reconstruction error, not ground-truth accuracy",
        "stride": stride,
        "frames": audits,
        "verified_changes": [],
        "limitations": "Predicted surfaces, not measured ground truth. Self-reprojection does not establish depth accuracy or object-level registration.",
    }
    (output / "diagnostics.json").write_text(json.dumps(summary, indent=2))
    return summary


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run',required=True,type=Path);p.add_argument('--output',required=True,type=Path);p.add_argument('--stride',type=int,default=2)
    args=p.parse_args();result=export_dense(args.run,args.output,args.stride)
    print(json.dumps({k:result[k] for k in ['before_points','after_points','scale']},indent=2))
