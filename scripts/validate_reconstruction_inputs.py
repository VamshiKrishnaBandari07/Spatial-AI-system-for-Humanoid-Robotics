"""Validate actual prepared native inputs without inference, training or new downloads."""

import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
from spatialmemory.public_data import DatasetSequence, load_frame, replica_atlas
from spatialmemory.rgbd_validation import (
    unproject,
    project,
    transform_points,
    depth_consistency,
)
from spatialmemory.exports import ply


def validate_sequence(path, output):
    sequence = DatasetSequence.model_validate_json(path.read_text())
    records, clouds = [], []
    previous = None
    comparisons = []
    for frame in sequence.frames:
        data = load_frame(frame)
        points, pixels = unproject(data["depth_metres"], data["K"], stride=8)
        world = transform_points(points, data["T_world_camera"])
        restored = transform_points(world, np.linalg.inv(data["T_world_camera"]))
        roundtrip = (
            float(np.max(abs(project(points, data["K"]) - pixels)))
            if len(points)
            else None
        )
        if (
            roundtrip is None
            or roundtrip > 1e-7
            or not np.allclose(points, restored, atol=1e-8)
        ):
            raise ValueError("Coordinate round trip failed or no valid depth")
        records.append(
            {
                "timestamp_ns": frame.timestamp_ns,
                "valid_sample_points": len(points),
                "pixel_roundtrip_max": roundtrip,
                "invalid_depth_pixels": int((~np.isfinite(data["depth_metres"])).sum()),
                "fx": float(data["K"][0, 0]),
                "K": data["K"].tolist(),
                "rgb_calibration_available": data["K_rgb"] is not None,
                "rgb_depth_aligned": data["rgb_depth_aligned"],
                "sync_error_seconds": frame.synchronization_error_seconds,
            }
        )
        if previous is not None and data["rgb_depth_aligned"]:
            comparisons.append(depth_consistency(previous, data))
        previous = data
        if sequence.dataset == "ARKitScenes":
            clouds.append(world)
    timestamps = [r["timestamp_ns"] for r in records]
    if any(a >= b for a, b in zip(timestamps, timestamps[1:])):
        raise ValueError("Sequence timestamps must increase")
    asset = None
    if clouds:
        asset = output / (sequence.recording_id + "_native_depth_world.ply")
        ply(asset, np.concatenate(clouds))
    return {
        "dataset": sequence.dataset,
        "scene_id": sequence.scene_id,
        "recording_id": sequence.recording_id,
        "source_manifest": str(path),
        "source_manifest_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "status": "passed",
        "frames_checked": len(records),
        "frames": records,
        "variable_intrinsics": len({tuple(np.asarray(r["K"]).ravel()) for r in records})
        > 1,
        "timestamp_source": sequence.timestamp_source,
        "cross_view_depth_diagnostics": comparisons,
        "point_cloud": str(asset) if asset else None,
        "metric_accuracy_evaluated": False,
        "notes": "Native-depth world cloud is sensor unprojection, not neural output. 7-Scenes nominal depth K is not RGB calibration; no aligned RGB-D accuracy is claimed.",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sequence", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--assets", type=Path, required=True)
    parser.add_argument("--replica-atlas", type=Path)
    args = parser.parse_args()
    args.assets.mkdir(parents=True, exist_ok=True)
    result = {"sequences": [validate_sequence(p, args.assets) for p in args.sequence]}
    if args.replica_atlas:
        atlas = replica_atlas(args.replica_atlas)
        result["replica"] = {
            "path": str(args.replica_atlas),
            "shape": list(atlas.shape),
            "finite": bool(np.isfinite(atlas).all()),
            "status": "appearance asset only",
            "complete_geometry": False,
        }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(
        json.dumps(
            {
                "sequences": [
                    {
                        "id": s["recording_id"],
                        "frames": s["frames_checked"],
                        "status": s["status"],
                    }
                    for s in result["sequences"]
                ],
                "replica": result.get("replica"),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
