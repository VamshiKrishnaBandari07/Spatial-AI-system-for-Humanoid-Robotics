import numpy as np
import pytest

from spatialmemory.metric_alignment import metric_register, alignment_metrics
from spatialmemory.sensor_geometry import rasterize_depth
from spatialmemory.rgbd_validation import transform_points


def test_metric_registration_recovers_sourced_rigid_transform():
    rng = np.random.default_rng(9)
    source = rng.uniform(-0.5, 0.5, (6000, 3))
    T = np.eye(4)
    T[:3, 3] = [0.08, -0.04, 0.02]
    target = transform_points(source, T)
    initial = T.copy()
    initial[:3, 3] += [0.005, -0.002, 0]
    result = metric_register(source, target, initial)
    assert result["valid"]
    assert result["refined"]["inlier_rmse_metres"] < 0.025
    assert (
        np.linalg.norm(np.asarray(result["T_reference_rescan"])[:3, 3] - T[:3, 3])
        < 0.01
    )
    assert result["heldout"]["source_points"] > 0


def test_metric_registration_rejects_nonoverlapping_clouds_without_fallback():
    p = np.random.default_rng(5).normal(size=(100, 3))
    result = metric_register(p, p + 10, np.eye(4))
    assert not result["valid"]
    assert result["refined"]["fitness"] == 0


def test_metric_transform_rejects_reflection_and_scale():
    p = np.random.default_rng(5).normal(size=(100, 3))
    bad = np.eye(4)
    bad[0, 0] = -1
    with pytest.raises(ValueError):
        metric_register(p, p, bad)
    bad[0, 0] = 2
    with pytest.raises(ValueError):
        metric_register(p, p, bad)


def test_depth_projection_keeps_holes_unknown_and_nearest_depth():
    depth = np.array([[1.0, 2.0], [np.nan, 0.0]])
    Kd = np.eye(3)
    Krgb = np.diag([0.1, 0.1, 1])
    projected = rasterize_depth(depth, Kd, Krgb, np.eye(4), 2, 2)
    assert projected[0, 0] == 1
    assert np.isnan(projected[1, 1])


def test_transform_direction_changes_alignment_metrics():
    p = np.random.default_rng(6).uniform(size=(1000, 3))
    T = np.eye(4)
    T[:3, 3] = [1, 0, 0]
    q = transform_points(p, T)
    assert alignment_metrics(p, q, T, 0.05)["fitness"] == 1
    assert alignment_metrics(p, q, np.linalg.inv(T), 0.05)["fitness"] == 0


def test_quality_selection_avoids_feature_poor_endpoints():
    from spatialmemory.contracts import Frame
    from spatialmemory.ingestion import neural_keyframes

    frames = [
        Frame(
            id=str(i),
            index=i,
            timestamp=float(i),
            path="unit-test.png",
            sharpness=100,
            exposure=0.5,
            features=count,
        )
        for i, count in enumerate([80, 5000, 3000, 80])
    ]
    assert [f.index for f in neural_keyframes(frames, 2)] == [1, 2]


def test_native_review_preserves_unknown_without_explicit_change(tmp_path):
    import json
    from spatialmemory.benchmarking import sha256
    from spatialmemory.native_review import review_native

    T = np.eye(4)
    T[0, 3] = 0.1
    relation = tmp_path / "relationship.json"
    relation.write_text(
        json.dumps(
            [
                {
                    "reference": "reference",
                    "type": "train",
                    "scans": [
                        {
                            "reference": "rescan",
                            "transform": T.flatten(order="F").tolist(),
                            "rigid": [
                                {
                                    "instance_reference": 1,
                                    "instance_rescan": 1,
                                    "transform": np.eye(4).flatten(order="F").tolist(),
                                }
                            ],
                        }
                    ],
                }
            ]
        )
    )
    for scan in ["reference", "rescan"]:
        directory = tmp_path / scan
        directory.mkdir()
        path = directory / "semseg.v2.json"
        path.write_text(
            json.dumps(
                {
                    "scan_id": scan,
                    "segGroups": [
                        {"objectId": i, "label": "box", "obb": {"centroid": [i, 0, 1]}}
                        for i in [1, 2]
                    ],
                }
            )
        )
        path.with_suffix(path.suffix + ".download.json").write_text(
            json.dumps({"sha256": sha256(path)})
        )
    review = review_native(
        {
            "relationship_metadata_path": str(relation),
            "relationship_metadata_sha256": sha256(relation),
            "reference": "reference",
            "rescan": "rescan",
            "raw_destination": str(tmp_path),
        }
    )
    assert review["objects"][0]["change_type"] == "MOVED"
    assert review["objects"][1]["change_type"] == "UNKNOWN"
    assert "ADDED" in review["unsupported_categories"]


def test_metric_pipeline_orchestration_keeps_neural_invocation_and_exports(
    tmp_path, monkeypatch
):
    """Mocked expert unit regression; the real model run is separate evidence."""
    import json
    from types import SimpleNamespace
    from spatialmemory.contracts import Config
    from spatialmemory.pipeline import run
    from spatialmemory.geometry import Reconstruction
    from spatialmemory.benchmarking import sha256
    from spatialmemory import neural_geometry, sensor_geometry
    from test_ingestion_pipeline import fixture

    before = fixture.make_video(tmp_path / "before.mp4")
    after = fixture.make_video(tmp_path / "after.mp4")
    relationship = tmp_path / "native.json"
    relationship.write_text(
        json.dumps(
            [
                {
                    "reference": "R",
                    "scans": [
                        {
                            "reference": "A",
                            "transform": np.eye(4).flatten(order="F").tolist(),
                        }
                    ],
                }
            ]
        )
    )
    seed = tmp_path / "seed.json"
    seed.write_text(
        json.dumps(
            {
                "unit": "metres",
                "source_frame": "rescan_world",
                "destination_frame": "reference_world",
                "relationship_file": str(relationship),
                "relationship_sha256": sha256(relationship),
                "before_recording_id": "R",
                "after_recording_id": "A",
                "before_video_sha256": sha256(before),
                "after_video_sha256": sha256(after),
                "T_reference_rescan": np.eye(4).tolist(),
            }
        )
    )
    calls = []
    monkeypatch.setattr(neural_geometry, "load_expert", lambda: object())

    def neural(*args):
        calls.append(True)
        return SimpleNamespace(diagnostics={"unit_test_mock": True})

    monkeypatch.setattr(neural_geometry, "reconstruct_neural", neural)
    points = np.random.default_rng(8).uniform(-0.5, 0.5, (6000, 3))
    monkeypatch.setattr(
        sensor_geometry,
        "reconstruct_sensor",
        lambda frames, files, diag: Reconstruction(
            points,
            np.ones_like(points) * 0.5,
            {},
            {},
            {},
            np.eye(3),
            {"sensor_geometry_used": True, "neural_cross_validation": diag},
        ),
    )
    report = run(
        before,
        after,
        tmp_path / "run",
        Config(
            metric_geometry_enabled=True,
            registration_initial=str(seed),
            geometry_backend="vggt",
        ),
    )
    assert len(calls) == 2
    assert report["status"] == "complete" and report["registration"]["valid"]
    assert (tmp_path / "run/reconstruction/aligned_scene.ply").exists()
    assert not report["events"]
