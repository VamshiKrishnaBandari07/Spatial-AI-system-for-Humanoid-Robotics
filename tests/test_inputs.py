import json, struct, zlib
import numpy as np, pytest, cv2
from spatialmemory.inputs import (
    Manifest,
    Recording,
    Camera,
    Annotations,
    Poses,
    DepthMetadata,
    read_depth,
    read_imu,
    load_manifest,
    resolve,
)
from spatialmemory.datasets import extract_scannet_sens, convert_3rscan, convert_scannet
from spatialmemory.memory import Observation, Memory, associate, scene_graph
from spatialmemory.tracking import track
from spatialmemory.evaluation import classification_metrics


def test_paths_do_not_escape(tmp_path):
    with pytest.raises(ValueError):
        resolve("../private.mp4", tmp_path)
    (tmp_path / "alias").symlink_to("/tmp")
    with pytest.raises(ValueError):
        resolve("alias/private.mp4", tmp_path)


def test_optional_missing_does_not_block(tmp_path):
    (tmp_path / "before.mp4").touch()
    (tmp_path / "after.mp4").touch()
    m = Manifest(
        scene_id="scene_001",
        before=Recording(video="before.mp4", camera="missing.json"),
        after=Recording(video="after.mp4"),
        annotations="missing_annotations.json",
    )
    path = tmp_path / "manifest.json"
    path.write_text(m.model_dump_json())
    _, r, w = load_manifest(path, tmp_path)
    assert r["before"]["camera"] is None and len(w) == 2


def test_camera_rescaling_and_annotation_validation():
    c = Camera(width=640, height=480, fx=600, fy=610, cx=320, cy=240, source="test")
    assert c.matrix(320, 240)[0, 0] == 300
    with pytest.raises(ValueError):
        Annotations(
            scene_id="s", position_unit="metres", scale_verified=False, changes=[]
        )
    a = Annotations(
        scene_id="s",
        changes=[
            {
                "object_id": "o",
                "label": "chair",
                "change_type": "MOVED",
                "annotation_source": "human",
            }
        ],
    )
    assert a.changes[0].before_position is None


def test_pose_validation():
    with pytest.raises(ValueError):
        Poses(
            translation_unit="metres",
            poses=[
                {
                    "timestamp_ns": 0,
                    "translation": [0, 0, 0],
                    "quaternion_xyzw": [0, 0, 0, 2],
                }
            ],
        )


def test_depth_formats(tmp_path):
    m = DepthMetadata(
        unit="millimetres",
        width=2,
        height=2,
        camera="camera.json",
        rgb_aligned=True,
        frames=[],
    )
    a = np.array([[1000, 0], [2000, 3000]], np.uint16)
    for ext in ["png", "tiff", "npy", "npz"]:
        path = tmp_path / f"d.{ext}"
        if ext == "npy":
            np.save(path, a)
        elif ext == "npz":
            np.savez(path, depth=a)
        else:
            cv2.imwrite(str(path), a)
        depth = read_depth(path, m)
        assert depth[0, 0] == 1 and np.isnan(depth[0, 1])


def test_imu_requires_units_and_sync(tmp_path):
    p = tmp_path / "imu.csv"
    p.write_text(
        "timestamp_ns,ax,ay,az,gx,gy,gz\n0,0,0,9.81,0,0,0\n100,0,0,9.81,0,0,0\n"
    )
    with pytest.raises(ValueError):
        read_imu(p)
    p.with_suffix(".metadata.json").write_text(
        json.dumps(
            {
                "acceleration_unit": "m/s^2",
                "angular_velocity_unit": "rad/s",
                "time_offset_ns": 0,
                "calibration": {"source": "fixture"},
            }
        )
    )
    assert read_imu(p)["measurements"].shape == (2, 7)


def test_native_scannet_sensor_decode(tmp_path):
    file = tmp_path / "fixture.sens"
    image = np.full((4, 4, 3), 100, np.uint8)
    _, encoded = cv2.imencode(".jpg", image)
    depth = zlib.compress(np.full((2, 2), 1000, np.uint16).tobytes())
    with file.open("wb") as f:
        f.write(struct.pack("<IQ", 4, 4))
        f.write(b"test")
        for _ in range(4):
            f.write(struct.pack("<16f", *np.eye(4).ravel()))
        f.write(struct.pack("<iiIIIIfQ", 2, 1, 4, 4, 2, 2, 1000.0, 1))
        f.write(struct.pack("<16f", *np.eye(4).ravel()))
        f.write(struct.pack("<QQQQ", 100, 102, len(encoded), len(depth)))
        f.write(encoded.tobytes())
        f.write(depth)
    result = extract_scannet_sens(file, tmp_path / "extracted")
    assert np.load(result / "depth/000000.npy")[0, 0] == 1
    assert (
        json.loads((result / "sensor_metadata.json").read_text())["timestamps"][0][
            "rgb_timestamp_us"
        ]
        == 100
    )


def test_dataset_pair_rejection(tmp_path):
    p = tmp_path / "pairs.json"
    p.write_text("[]")
    with pytest.raises(ValueError):
        convert_3rscan(tmp_path, p, "a", "b", tmp_path / "out", "train")
    p.write_text("{}")
    with pytest.raises(ValueError):
        convert_scannet(tmp_path, "a", "b", p, tmp_path / "out", "train")


def test_memory_and_identical_identity_ambiguity(tmp_path):
    o = lambda name: Observation(
        id=name,
        frame_id="f",
        timestamp=0,
        label="chair",
        descriptor=[1, 0],
        viewpoint="v",
        position=[0, 0, 0],
    )
    before = [o("a"), o("b")]
    after = [o("c"), o("d")]
    assert all(not a["identity_verified"] for a in associate(before, after))
    memory = Memory(tmp_path / "memory.db")
    memory.observe(before[0])
    memory.snapshot("before", {"objects": []})
    assert len(memory.observations()) == 1 and memory.world("before") == {"objects": []}
    with pytest.raises(Exception):
        memory.snapshot("before", {})
    memory.close()
    assert scene_graph(before)["edges"][0]["relation"] == "NEAR"


def test_flow_translation_is_not_physical_movement():
    rng = np.random.default_rng(8)
    a = rng.integers(0, 256, (100, 100, 3), dtype=np.uint8)
    b = cv2.warpAffine(a, np.array([[1, 0, 2], [0, 1, 0]], float), (100, 100))
    points = np.array([[x, y] for x in range(20, 80, 10) for y in range(20, 80, 10)])
    result = track(a, b, points)
    assert sum(result["valid"]) >= 20 and not result["motion_is_physical"]


def test_measured_metrics():
    m = classification_metrics(
        ["ADDED", "REMOVED", "MOVED"], ["ADDED", "UNKNOWN", "MOVED"]
    )
    assert m["accuracy"] == 2 / 3 and m["per_class"]["REMOVED"]["recall"] == 0
    with pytest.raises(ValueError):
        classification_metrics([], [])


def test_3rscan_conversion_preserves_native_pair_and_split(tmp_path):
    from spatialmemory.datasets import convert_3rscan

    root = tmp_path / "raw"
    for scan in ["reference", "rescan"]:
        seq = root / scan / "sequence"
        seq.mkdir(parents=True)
        K = np.eye(4)
        K[0, 0] = K[1, 1] = 60
        K[0, 2] = K[1, 2] = 32
        (seq / "_info.txt").write_text(
            "m_colorWidth = 64\nm_colorHeight = 64\nm_calibrationColorIntrinsic = "
            + " ".join(map(str, K.ravel()))
        )
        for i in range(3):
            cv2.imwrite(
                str(seq / f"frame-{i:06d}.color.jpg"),
                np.full((64, 64, 3), i * 40, np.uint8),
            )
            np.savetxt(seq / f"frame-{i:06d}.pose.txt", np.eye(4))
    pairs = tmp_path / "pairs.json"
    pairs.write_text(
        json.dumps([{"reference": "reference", "scans": [{"reference": "rescan"}]}])
    )
    path = convert_3rscan(
        root, pairs, "reference", "rescan", tmp_path / "canonical", "validation"
    )
    m, r, w = load_manifest(path, tmp_path)
    assert m.dataset["split"] == "validation" and r["before"]["camera_data"].fx == 60
    assert len(r["after"]["pose_data"].poses) == 3


def test_model_revision_guard_and_explicit_unloaded_failure(tmp_path):
    from spatialmemory.models import Expert, ModelSpec

    with pytest.raises(ValueError):
        Expert(ModelSpec("facebook/dinov2-small", "main", "appearance"), tmp_path)
    expert = Expert(
        ModelSpec("facebook/dinov2-small", "0" * 40, "appearance"), tmp_path
    )
    with pytest.raises(RuntimeError):
        expert.infer(None)


def test_pose_arrays_and_rotation_matrix(tmp_path):
    from spatialmemory.inputs import read_poses, Pose

    meta = {
        "pose_convention": "T_world_camera",
        "translation_unit": "metres",
        "source": "fixture",
    }
    arr = np.array([[0, 0, 0, 0, 0, 0, 0, 1], [100, 1, 0, 0, 0, 0, 0, 1.0]])
    for ext in ["txt", "npy"]:
        p = tmp_path / f"poses.{ext}"
        p.with_suffix(".metadata.json").write_text(json.dumps(meta))
        if ext == "txt":
            np.savetxt(p, arr)
        else:
            np.save(p, arr)
        assert len(read_poses(p).poses) == 2
    pose = Pose(
        timestamp_ns=0, translation=[0, 0, 0], rotation_matrix=np.eye(3).tolist()
    )
    assert pose.quaternion_xyzw == [0, 0, 0, 1]


def test_sensor_timestamp_mismatch():
    from spatialmemory.inputs import validate_sensor_timing

    m = DepthMetadata(
        unit="metres",
        width=1,
        height=1,
        camera="camera.json",
        rgb_aligned=True,
        frames=[{"file": "d.npy", "timestamp_ns": 1000000000, "rgb_frame_index": 0}],
    )
    with pytest.raises(ValueError):
        validate_sensor_timing(
            {"depth_metadata": m}, {"original_timestamps_seconds": [0, 0.1]}
        )


def test_evaluation_missing_ground_truth_is_clear(tmp_path):
    import subprocess, sys

    manifest = Manifest(
        scene_id="s",
        before=Recording(video="missing_before.mp4"),
        after=Recording(video="missing_after.mp4"),
    )
    path = tmp_path / "manifest.json"
    path.write_text(manifest.model_dump_json())
    p = subprocess.run(
        [sys.executable, "-m", "spatialmemory_x", "evaluate", "--manifest", str(path)],
        capture_output=True,
        text=True,
    )
    assert p.returncode == 2 and "requires real ground-truth" in p.stderr


def test_3rscan_native_depth_conversion(tmp_path):
    root = tmp_path / "raw"
    K = np.eye(4)
    K[0, 0] = K[1, 1] = 30
    K[0, 2] = K[1, 2] = 16
    for name in ["ref", "scan"]:
        seq = root / name / "sequence"
        seq.mkdir(parents=True)
        (seq / "_info.txt").write_text(
            "\n".join(
                f"{key} = {value}"
                for key, value in {
                    "m_colorWidth": 32,
                    "m_colorHeight": 32,
                    "m_depthShift": 1000,
                    "m_calibrationColorIntrinsic": " ".join(map(str, K.ravel())),
                    "m_calibrationDepthIntrinsic": " ".join(map(str, K.ravel())),
                    "m_calibrationColorExtrinsic": " ".join(
                        map(str, np.eye(4).ravel())
                    ),
                    "m_calibrationDepthExtrinsic": " ".join(
                        map(str, np.eye(4).ravel())
                    ),
                }.items()
            )
        )
        for i in range(2):
            cv2.imwrite(
                str(seq / f"frame-{i:06d}.color.jpg"),
                np.full((32, 32, 3), 60, np.uint8),
            )
            cv2.imwrite(
                str(seq / f"frame-{i:06d}.depth.pgm"),
                np.full((32, 32), 1000, np.uint16),
            )
    pairs = tmp_path / "pairs.json"
    pairs.write_text(
        json.dumps([{"reference": "ref", "scans": [{"reference": "scan"}]}])
    )
    path = convert_3rscan(root, pairs, "ref", "scan", tmp_path / "canonical", "test")
    _, r, _ = load_manifest(path, tmp_path)
    assert r["before"]["depth_metadata"].scale_verified
    assert np.load(r["before"]["depth_dir"] / "000000.npy")[0, 0] == 1
