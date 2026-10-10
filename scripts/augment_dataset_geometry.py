"""Controlled geometric-region examples from verified metric RGB-D, not real object annotations."""

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from spatialmemory.inputs import Annotations, Annotation, Manifest, Camera
from spatialmemory.public_data import (
    DatasetSequence,
    SequenceFrame,
    load_frame,
    export_recording,
)


def render(points, colors, identities, pose, camera):
    camera_points = (points - pose[:3, 3]) @ pose[:3, :3]
    z = camera_points[:, 2]
    valid = z > 0.05
    index = np.flatnonzero(valid)
    uv = camera_points[valid] @ camera.matrix(camera.width, camera.height).T
    pixels = np.rint(uv[:, :2] / uv[:, 2, None]).astype(int)
    inside = (
        (pixels[:, 0] >= 0)
        & (pixels[:, 0] < camera.width)
        & (pixels[:, 1] >= 0)
        & (pixels[:, 1] < camera.height)
    )
    index = index[inside]
    pixels = pixels[inside]
    if len(index) == 0:
        return (
            np.zeros((camera.height, camera.width, 3), np.uint8),
            np.zeros((camera.height, camera.width), np.float32),
            np.zeros((camera.height, camera.width), np.uint8),
        )
    flat = pixels[:, 1] * camera.width + pixels[:, 0]
    order = np.lexsort((z[index], flat))
    ordered = flat[order]
    selected = order[np.r_[True, ordered[1:] != ordered[:-1]]]
    index = index[selected]
    pixels = pixels[selected]
    image = np.zeros((camera.height, camera.width, 3), np.uint8)
    depth = np.zeros((camera.height, camera.width), np.float32)
    labels = np.zeros((camera.height, camera.width), np.uint8)
    image[pixels[:, 1], pixels[:, 0]] = np.clip(colors[index] * 255, 0, 255).astype(
        np.uint8
    )
    depth[pixels[:, 1], pixels[:, 0]] = z[index]
    labels[pixels[:, 1], pixels[:, 0]] = identities[index]
    return image, depth, labels


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sequence", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    source = DatasetSequence.model_validate_json(args.sequence.read_text())
    if source.split != "train" or not all(
        f.rgb_depth_aligned and f.calibration_kind == "rgb" for f in source.frames
    ):
        raise ValueError(
            "Augment only registered calibrated training scenes; never leak a held-out scene"
        )
    args.output.mkdir(parents=True, exist_ok=False)
    selected = np.unique(
        np.linspace(0, len(source.frames) - 1, min(8, len(source.frames)))
        .round()
        .astype(int)
    )
    points = []
    colors = []
    views = []
    for index in selected:
        frame = source.frames[index]
        data = load_frame(frame)
        views.append(data["T_world_camera"])
        y, x = np.mgrid[0 : frame.camera.height : 3, 0 : frame.camera.width : 3]
        z = data["depth_metres"][y, x]
        valid = np.isfinite(z) & (z > 0)
        y = y[valid]
        x = x[valid]
        z = z[valid]
        camera = np.linalg.inv(data["K"]) @ np.vstack([x * z, y * z, z])
        world = (
            data["T_world_camera"][:3, :3] @ camera
            + data["T_world_camera"][:3, 3, None]
        ).T
        points.append(world)
        colors.append(data["rgb"][:, y, x].T)
    points = np.vstack(points)
    colors = np.vstack(colors)
    center = points[len(points) // 2]
    region = np.linalg.norm(points - center, axis=1) < 0.20
    if region.sum() < 20:
        raise ValueError("Insufficient sampled region geometry")
    center = points[region].mean(axis=0)
    ids = np.where(region, 1, 0).astype(np.uint8)
    native_camera = source.frames[0].camera
    camera = Camera(
        **{
            **native_camera.model_dump(),
            "source": "Synthetic renderer fixed K from first native ARKit calibration; not an original sensor recording",
        }
    )
    cases = [
        ("unchanged", "UNCHANGED", 0),
        ("moved", "MOVED", 0.20),
        ("small_displacement", "MOVED", 0.02),
        ("added", "ADDED", 0.40),
        ("removed", "REMOVED", 0),
        ("lighting", "UNCHANGED", 0),
        ("occlusion", "UNCHANGED", 0),
    ]
    entries = []
    for name, change, offset in cases:
        folder = args.output / name
        folder.mkdir()
        after_points = points.copy()
        after_colors = colors.copy()
        after_ids = ids.copy()
        delta = np.array([offset, 0, 0])
        if change == "MOVED":
            after_points[region] += delta
        elif change == "REMOVED":
            after_points = points[~region]
            after_colors = colors[~region]
            after_ids = ids[~region]
        elif change == "ADDED":
            after_points = np.vstack([points, points[region] + delta])
            after_colors = np.vstack([colors, colors[region]])
            after_ids = np.r_[ids, np.full(region.sum(), 2, np.uint8)]
        before_visible = after_visible = 0
        recordings = {}
        for state, p, c, identity in [
            ("before", points, colors, ids),
            ("after", after_points, after_colors, after_ids),
        ]:
            raw = folder / (state + "_rendered")
            raw.mkdir()
            frames = []
            for index, pose in enumerate(views):
                image, depth, labels = render(p, c, identity, pose, camera)
                if state == "after" and name == "lighting":
                    image = np.clip(image.astype(float) * 0.8, 0, 255).astype(np.uint8)
                if state == "after" and name == "occlusion":
                    y0, y1 = camera.height // 3, 2 * camera.height // 3
                    x0, x1 = camera.width // 3, 2 * camera.width // 3
                    image[y0:y1, x0:x1] = 0
                    depth[y0:y1, x0:x1] = 0
                    labels[y0:y1, x0:x1] = 0
                if state == "before":
                    before_visible += int(np.count_nonzero(labels == 1))
                else:
                    after_visible += int(
                        np.count_nonzero(labels == (2 if change == "ADDED" else 1))
                    )
                rgb = raw / f"{index:06d}.png"
                dep = raw / f"{index:06d}.npy"
                cv2.imwrite(str(rgb), cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
                np.save(dep, depth)
                np.save(raw / f"{index:06d}.region_ids.npy", labels)
                frames.append(
                    SequenceFrame(
                        rgb=str(rgb.resolve()),
                        depth=str(dep.resolve()),
                        timestamp_ns=round(index / 30 * 1e9),
                        source_frame=f"synthetic_{index}",
                        camera=camera,
                        T_world_camera=pose.tolist(),
                        depth_unit="metres",
                        synchronization_error_seconds=0,
                    )
                )
            sequence = DatasetSequence(
                dataset="synthetic_ARKitScenes",
                scene_id=source.scene_id,
                recording_id=name + "_" + state,
                split="train",
                native_split=source.native_split,
                source=str(args.sequence.resolve()),
                license=source.license,
                timestamp_source="synthetic encoding schedule; source native camera transforms retained",
                frames=frames,
            )
            recordings[state] = export_recording(sequence, folder / state)
        annotations = [
            Annotation(
                object_id="synthetic_region_added"
                if change == "ADDED"
                else "synthetic_region_1",
                label="synthetic geometric region, not verified semantic object",
                change_type=change,
                before_position=None if change == "ADDED" else center.tolist(),
                after_position=None
                if change == "REMOVED"
                else (center + delta).tolist(),
                visibility="OBSERVED"
                if (after_visible if change == "ADDED" else before_visible) > 0
                else "UNKNOWN",
                annotation_source="Exact synthetic point-set transformation; not real-world changes",
            )
        ]
        scene = f"synthetic_arkit_{source.recording_id}_{name}"
        ground_truth = Annotations(
            scene_id=scene,
            coordinate_frame="native ARKit world",
            position_unit="metres",
            scale_verified=True,
            changes=annotations,
        )
        annotation_path = folder / "changes.json"
        annotation_path.write_text(ground_truth.model_dump_json(indent=2))
        transform = {
            "source_scene": source.scene_id,
            "source_sequence": str(args.sequence.resolve()),
            "split": "train",
            "region_definition": "Euclidean radius 0.20m around selected source point; geometric fragment, not object segmentation",
            "point_count_before": len(points),
            "region_point_count": int(region.sum()),
            "translation_metres": delta.tolist(),
            "before_visible_region_pixels": before_visible,
            "after_visible_region_pixels": after_visible,
            "lighting_multiplier": 0.8 if name == "lighting" else 1,
            "occlusion_rectangle_xywh": [
                camera.width // 3,
                camera.height // 3,
                camera.width // 3,
                camera.height // 3,
            ]
            if name == "occlusion"
            else None,
            "renderer": "Nearest point depth buffer; unfilled pixels invalid depth 0, not free-space proof",
            "geometry_completeness": "Sparse sampled source surfaces; no hidden geometry or semantic object truth invented",
            "actual_real_world_change": False,
        }
        (folder / "transformation.json").write_text(json.dumps(transform, indent=2))
        manifest = Manifest(
            scene_id=scene,
            before=recordings["before"],
            after=recordings["after"],
            annotations=str(annotation_path.resolve()),
            rgb_only=False,
            dataset={"name": "synthetic_ARKitScenes", **transform},
        )
        (folder / "manifest.json").write_text(manifest.model_dump_json(indent=2))
        entries.append(str((folder / "manifest.json").resolve()))
    (args.output / "examples.json").write_text(json.dumps(entries, indent=2))
    print(
        json.dumps(
            {
                "synthetic_examples": len(entries),
                "source_scene": source.scene_id,
                "split": "train",
            }
        )
    )


if __name__ == "__main__":
    main()
