import zipfile
from pathlib import Path

import cv2
import numpy as np
import pytest

from spatialmemory.inputs import Camera, Manifest, load_manifest
from spatialmemory.public_data import (
    arkit_sequence,
    seven_scenes_sequence,
    export_recording,
    load_frame,
    write_splits,
    replica_atlas,
    extract_zip,
    valid_transform,
)


def native_arkit(root):
    for folder in ["lowres_wide", "lowres_depth", "lowres_wide_intrinsics"]:
        (root / folder).mkdir(parents=True)
    for i in range(3):
        stem = f"123_{i * 0.1:.3f}"
        cv2.imwrite(
            str(root / "lowres_wide" / (stem + ".png")),
            np.full((48, 64, 3), 100, np.uint8),
        )
        cv2.imwrite(
            str(root / "lowres_depth" / (stem + ".png")),
            np.full((48, 64), 2000, np.uint16),
        )
        (root / "lowres_wide_intrinsics" / (stem + ".pincam")).write_text(
            "64 48 50 50 32 24"
        )
    (root / "lowres_wide.traj").write_text(
        "\n".join(f"{i * 0.1} 0 0 0 {i * 0.2} 0 0" for i in range(3))
    )


def test_arkit_pose_inversion_and_canonical_readers(tmp_path):
    root = tmp_path / "native"
    native_arkit(root)
    sequence = arkit_sequence(root, "visit1", "123", "train")
    data = load_frame(sequence.frames[1])
    assert data["rgb"].shape == (3, 48, 64) and data["depth_metres"][0, 0] == 2
    assert data["T_world_camera"][0, 3] == pytest.approx(-0.2)
    recording = export_recording(sequence, tmp_path / "canonical")
    manifest = Manifest(
        scene_id="control",
        before=recording,
        after=recording,
        rgb_only=False,
        dataset={
            "task": "explicit same-recording loader control, no change ground truth"
        },
    )
    path = tmp_path / "manifest.json"
    path.write_text(manifest.model_dump_json())
    _, resolved, warnings = load_manifest(path, tmp_path)
    assert not warnings and len(resolved["before"]["pose_data"].poses) == 3
    assert resolved["before"]["depth_metadata"].scale_verified


def test_arkit_rejects_unsynchronized_poses(tmp_path):
    native_arkit(tmp_path)
    (tmp_path / "lowres_wide.traj").write_text("2 0 0 0 0 0 0\n3 0 0 0 0 0 0")
    with pytest.raises(ValueError, match="synchronized"):
        arkit_sequence(tmp_path, "s", "r", "train")


def test_scene_split_leakage_is_rejected(tmp_path):
    root = tmp_path / "raw"
    native_arkit(root)
    sequence = arkit_sequence(root, "visit1", "123", "train")
    a = tmp_path / "a.json"
    a.write_text(sequence.model_dump_json())
    b = tmp_path / "b.json"
    b.write_text(sequence.model_copy(update={"split": "test"}).model_dump_json())
    with pytest.raises(ValueError, match="leaks"):
        write_splits([a, b], tmp_path / "splits")
    result = write_splits([a], tmp_path / "good")
    assert len(result["train"]) == 1 and result["test"] == []


def test_seven_scenes_invalid_depth_and_metric_pose(tmp_path):
    for i in range(3):
        stem = f"frame-{i:06d}"
        cv2.imwrite(
            str(tmp_path / (stem + ".color.png")), np.zeros((48, 64, 3), np.uint8)
        )
        depth = np.full((48, 64), 1000, np.uint16)
        depth[0, 0] = 65535
        cv2.imwrite(str(tmp_path / (stem + ".depth.png")), depth)
        pose = np.eye(4)
        pose[0, 3] = i * 0.1
        np.savetxt(tmp_path / (stem + ".pose.txt"), pose)
    camera = Camera(
        width=64,
        height=48,
        fx=50,
        fy=50,
        cx=32,
        cy=24,
        source="synthetic test calibration",
    )
    sequence = seven_scenes_sequence(tmp_path, "chess", "seq01", "train", camera)
    data = load_frame(sequence.frames[1])
    assert np.isnan(data["depth_metres"][0, 0]) and data["depth_metres"][1, 1] == 1
    assert data["T_world_camera"][0, 3] == pytest.approx(0.1)
    assert sequence.change_annotations is None
    assert data["K_rgb"] is None and not data["rgb_depth_aligned"]
    recording = export_recording(sequence, tmp_path / "canonical")
    assert recording.camera is None and recording.depth_dir is None
    assert (tmp_path / "canonical/depth_camera_nominal.json").exists()


def test_unsafe_zip_and_invalid_transform_rejected(tmp_path):
    path = tmp_path / "bad.zip"
    with zipfile.ZipFile(path, "w") as zipped:
        zipped.writestr("../escape", "data")
    with pytest.raises(ValueError, match="Unsafe"):
        extract_zip(path, tmp_path / "out")
    with pytest.raises(ValueError, match="orthonormal"):
        valid_transform(np.diag([2, 1, 1, 1]))
    assert not (tmp_path / "escape").exists()


def test_native_replica_half_float_atlas(tmp_path):
    path = tmp_path / "color-ptex.hdr"
    array = np.arange(48, dtype="<f2").reshape(4, 4, 3)
    array.tofile(path)
    loaded = replica_atlas(path)
    assert np.array_equal(array, loaded)
    path.write_bytes(b"bad")
    with pytest.raises(ValueError, match="size"):
        replica_atlas(path)


def test_geometric_renderer_depth_buffer_and_unknown_holes():
    import importlib.util

    path = Path(__file__).parents[1] / "scripts/augment_dataset_geometry.py"
    spec = importlib.util.spec_from_file_location("dataset_augmentation", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    camera = Camera(
        width=64, height=48, fx=50, fy=50, cx=32, cy=24, source="synthetic test"
    )
    points = np.array([[0, 0, 2], [0, 0, 1], [0, 0, -1]])
    rgb, depth, labels = module.render(
        points,
        np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]]),
        np.array([1, 2, 3]),
        np.eye(4),
        camera,
    )
    assert (
        depth[24, 32] == 1 and labels[24, 32] == 2 and list(rgb[24, 32]) == [0, 255, 0]
    )
    assert depth[0, 0] == 0 and labels[0, 0] == 0
    _, empty_depth, _ = module.render(
        points[2:], np.ones((1, 3)), np.array([3]), np.eye(4), camera
    )
    assert not empty_depth.any()


def test_augmented_source_scene_cannot_leak_into_holdout(tmp_path):
    native_arkit(tmp_path / "raw")
    source = arkit_sequence(tmp_path / "raw", "shared_visit", "recording", "train")
    original = tmp_path / "original.json"
    original.write_text(source.model_dump_json())
    augmented = tmp_path / "augmented.json"
    augmented.write_text(
        source.model_copy(
            update={"dataset": "synthetic_ARKitScenes", "split": "test"}
        ).model_dump_json()
    )
    with pytest.raises(ValueError, match="leaks"):
        write_splits([original, augmented], tmp_path / "splits")


def test_repeat_extraction_checks_integrity_without_overwrite(tmp_path):
    from spatialmemory.public_data import verify_extracted_zip

    archive = tmp_path / "native.zip"
    with zipfile.ZipFile(archive, "w") as zipped:
        zipped.writestr("seq/frame.pose.txt", "native bytes")
        zipped.writestr("seq/Thumbs.db", "not a dataset modality")
    output = tmp_path / "raw"
    extract_zip(archive, output, exclude_names=("Thumbs.db",))
    assert verify_extracted_zip(archive, output, exclude_names=("Thumbs.db",)) == 1
    assert not (output / "seq/Thumbs.db").exists()
    (output / "seq/frame.pose.txt").write_text("changed data")
    with pytest.raises(ValueError, match="CRC mismatch"):
        verify_extracted_zip(archive, output, exclude_names=("Thumbs.db",))
    assert (output / "seq/frame.pose.txt").read_text() == "changed data"
