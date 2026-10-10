"""Explicit synthetic/unit fixtures test guards and metric arithmetic, not accuracy."""

import copy
import json
import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient
from spatialmemory.inputs import Annotation, Annotations, Manifest, Recording
from spatialmemory.benchmarking import (
    PairEvidence,
    IdentityMapping,
    validate_pair,
    benchmark,
    sha256,
)
from spatialmemory.rgbd_validation import (
    unproject,
    project,
    transform_points,
    depth_consistency,
)
from spatialmemory.app import app


def test_depth_projection_variable_intrinsics_and_invalid_depth():
    depth = np.array([[1, 0, np.nan], [np.inf, -1, 2]], dtype=float)
    K = np.array([[100, 0, 1], [0, 110, 1], [0, 0, 1]], dtype=float)
    points, pixels = unproject(depth, K)
    assert len(points) == 2 and np.allclose(project(points, K), pixels)
    other = K.copy()
    other[0, 0] *= 2
    changed, _ = unproject(depth, other)
    assert not np.allclose(points, changed) and np.allclose(
        project(changed, other), pixels
    )


def test_nontrivial_camera_world_pose_roundtrip():
    T = np.array(
        [[0, -1, 0, 2], [1, 0, 0, -3], [0, 0, 1, 1], [0, 0, 0, 1]], dtype=float
    )
    points = np.array([[1, 2, 3], [-2, 1, 4]], dtype=float)
    assert np.allclose(
        transform_points(transform_points(points, T), np.linalg.inv(T)), points
    )


def test_visibility_depth_residual_excludes_occluded_and_unregistered():
    base = {
        "depth_metres": np.ones((3, 3)),
        "K": np.eye(3),
        "T_world_camera": np.eye(4),
        "rgb_depth_aligned": True,
    }
    source = dict(base, depth_metres=np.full((3, 3), 2.0))
    result = depth_consistency(source, base, stride=1)
    assert result["occluded_samples_excluded"] == 9 and result["usable_samples"] == 0
    assert result["median_absolute_depth_residual_metres"] is None
    with pytest.raises(ValueError, match="aligned"):
        depth_consistency(dict(base, rgb_depth_aligned=False), base)


def test_annotation_levels_and_independent_review():
    record = dict(
        object_id="unit-object",
        label="box",
        change_type="MOVED",
        annotation_source="explicit unit fixture",
    )
    assert Annotation(**record).annotation_level == 1
    with pytest.raises(ValueError, match="boxes or masks"):
        Annotation(**record, annotation_level=2)
    with pytest.raises(ValueError, match="measured 3D"):
        Annotation(**record, annotation_level=3)
    assert Annotation(
        **record, annotation_level=3, before_position=[1, 2, 3]
    ).before_position == [1, 2, 3]
    with pytest.raises(ValueError, match="distinct"):
        Annotation(
            **record,
            annotation_quality="INDEPENDENTLY_REVIEWED",
            annotator="A",
            independent_reviewer=" a ",
            reviewed_at="2026-10-10T00:00:00Z",
        )
    with pytest.raises(ValueError, match="timezone"):
        Annotation(**record, reviewed_at="2026-10-10T00:00:00")
    with pytest.raises(ValueError, match="measured displacement"):
        Annotation(
            **record,
            annotation_level=4,
            before_position=[1, 2, 3],
            before_geometry="unit.ply",
        )


def test_annotation_invalid_observation_and_metric_scale_rejected():
    from spatialmemory.inputs import AnnotationObservation

    with pytest.raises(ValueError, match="positive area"):
        AnnotationObservation(
            recording_id="unit",
            frame_id="0",
            bbox_xyxy=[2, 0, 1, 1],
            source="unit fixture",
        )
    with pytest.raises(ValueError, match="pre-existing"):
        Annotation(
            object_id="unit",
            label="unit",
            change_type="ADDED",
            before_position=[0, 0, 1],
            annotation_source="unit fixture",
        )
    with pytest.raises(ValueError, match="verified scale"):
        Annotations(scene_id="unit", position_unit="metres", changes=[])


def create_video(path, value):
    out = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 5, (64, 48))
    assert out.isOpened()
    for _ in range(3):
        out.write(np.full((48, 64, 3), value, np.uint8))
    out.release()


def pair_fixture(tmp_path):
    before = tmp_path / "before.mp4"
    after = tmp_path / "after.mp4"
    create_video(before, 10)
    create_video(after, 100)
    path = tmp_path / "manifest.json"
    path.write_text(
        Manifest(
            scene_id="unit",
            before=Recording(video=str(before)),
            after=Recording(video=str(after)),
        ).model_dump_json()
    )
    relation = tmp_path / "relation.json"
    relation.write_text(
        json.dumps(
            [{"reference": "unit-before", "scans": [{"reference": "unit-after"}]}]
        )
    )
    evidence = PairEvidence(
        dataset="unit fixture",
        before_recording_id="unit-before",
        after_recording_id="unit-after",
        source_url="https://example.invalid/unit-fixture",
        relationship_file=str(relation),
        relationship_sha256=sha256(relation),
        relationship_format="3RScan",
        data_origin="SYNTHETIC",
        documentation_note="Synthetic relationship-format fixture, not an official real pair",
    )
    return path, evidence


def test_pair_requires_actual_relationship_and_matching_registered_output(tmp_path):
    path, evidence = pair_fixture(tmp_path)
    pair = validate_pair(path, evidence, tmp_path)
    assert pair["pair_validity_status"] == "RELATIONSHIP_ONLY"
    assert (
        not pair["ground_truth_change_available"] and not pair["registration_verified"]
    )
    unrelated = evidence.model_copy(update={"after_recording_id": "unrelated-room"})
    assert (
        validate_pair(path, unrelated, tmp_path)["pair_validity_status"] == "REJECTED"
    )
    report = {
        "scene_id": "unit",
        "inputs": [],
        "registration": {"valid": True, "heldout_count": 10, "heldout_inliers": 10},
    }
    assert not validate_pair(path, evidence, tmp_path, report=report)[
        "registration_verified"
    ]


def test_pair_tampering_and_identical_recordings_rejected(tmp_path):
    path, evidence = pair_fixture(tmp_path)
    with pytest.raises(ValueError, match="checksum"):
        validate_pair(
            path,
            evidence.model_copy(update={"relationship_sha256": "0" * 64}),
            tmp_path,
        )
    manifest = json.loads(path.read_text())
    manifest["after"] = manifest["before"]
    path.write_text(json.dumps(manifest))
    assert validate_pair(path, evidence, tmp_path)["pair_validity_status"] == "REJECTED"


def test_provenance_gate_requires_source_changes_and_independent_annotations(tmp_path):
    path, evidence = pair_fixture(tmp_path)
    relation = tmp_path / "relation.json"
    relation.write_text(
        json.dumps(
            [
                {
                    "reference": "unit-before",
                    "scans": [{"reference": "unit-after", "removed": ["unit-object"]}],
                }
            ]
        )
    )
    evidence = evidence.model_copy(
        update={
            "relationship_sha256": sha256(relation),
            "physical_changes_documented": True,
            "data_origin": "REAL",
        }
    )
    # Fictional REAL-format records exercise validation only; never exported as real annotations.
    annotations = Annotations(
        scene_id="unit",
        data_origin="REAL",
        provenance="fictional unit fixture",
        changes=[
            Annotation(
                object_id="unit-object",
                label="unit",
                change_type="REMOVED",
                annotation_source="fictional unit fixture",
                generation_method="human",
                annotation_quality="INDEPENDENTLY_REVIEWED",
                annotator="unit-A",
                independent_reviewer="unit-B",
                reviewed_at="2026-10-10T00:00:00Z",
            )
        ],
    )
    manifest = Manifest.model_validate_json(path.read_text())
    report = {
        "scene_id": "unit",
        "status": "complete",
        "inputs": [
            {"sha256": sha256(manifest.before.video)},
            {"sha256": sha256(manifest.after.video)},
        ],
        "registration": {"valid": True, "heldout_count": 10, "heldout_inliers": 10},
    }
    assert (
        validate_pair(path, evidence, tmp_path, annotations, report)[
            "pair_validity_status"
        ]
        == "VERIFIED_FOR_EVALUATION"
    )
    unsafe = annotations.model_copy(deep=True)
    unsafe.changes[0].generation_method = "model_generated"
    assert (
        validate_pair(path, evidence, tmp_path, unsafe, report)["pair_validity_status"]
        == "RELATIONSHIP_ONLY"
    )
    relation.write_text(
        json.dumps(
            [{"reference": "unit-before", "scans": [{"reference": "unit-after"}]}]
        )
    )
    missing = evidence.model_copy(update={"relationship_sha256": sha256(relation)})
    assert not validate_pair(path, missing, tmp_path, annotations, report)[
        "physical_changes_documented"
    ]


def test_synthetic_origin_and_dataset_mismatch_do_not_pass_real_gate(tmp_path):
    path, evidence = pair_fixture(tmp_path)
    manifest = Manifest.model_validate_json(path.read_text())
    manifest.dataset = {"name": "wrong"}
    path.write_text(manifest.model_dump_json())
    with pytest.raises(ValueError, match="dataset"):
        validate_pair(path, evidence, tmp_path)
    manifest.dataset = {"data_origin": "SYNTHETIC"}
    path.write_text(manifest.model_dump_json())
    with pytest.raises(ValueError, match="Synthetic"):
        validate_pair(
            path, evidence.model_copy(update={"data_origin": "REAL"}), tmp_path
        )


def metric_fixture():
    # These artificial labels exercise metric arithmetic only. No benchmark artifact is produced.
    annotations = Annotations(
        scene_id="unit",
        exhaustive_object_inventory=True,
        changes=[
            Annotation(
                object_id="a",
                label="unit",
                change_type="MOVED",
                annotation_source="unit fixture",
            ),
            Annotation(
                object_id="b",
                label="unit",
                change_type="UNCHANGED",
                annotation_source="unit fixture",
            ),
        ],
    )
    report = {
        "scene_id": "unit",
        "events": [
            {"id": "event-a", "object_id": "pred-a", "change": "MOVED"},
            {"id": "event-b", "object_id": "pred-b", "change": "REMOVED"},
            {"id": "extra", "object_id": "pred-extra", "change": "ADDED"},
        ],
        "registration": {"rmse": 0.01},
        "config": {"geometry_backend": "vggt"},
    }
    mapping = IdentityMapping(
        scene_id="unit",
        ground_truth_to_prediction={"a": "pred-a", "b": "pred-b"},
        provenance="unit fixture association",
        verified_by="unit reviewer",
    )
    pair = {
        "pair_validity_status": "VERIFIED_FOR_EVALUATION",
        "scene_id": "unit",
        "source_dataset": "unit fixture",
    }
    return report, annotations, pair, mapping


def test_metrics_include_unannotated_physical_predictions_and_explicit_denominators():
    report, annotations, pair, mapping = metric_fixture()
    result = benchmark(report, annotations, pair, mapping)
    assert result["accuracy"] == 0.5 and result[
        "physical_change_precision"
    ] == pytest.approx(1 / 3)
    assert result["physical_change_recall"] == 1 and result["false_positive_rate"] == 1
    assert result["false_removal_rate"] == 1 and result["peak_memory_mib"] is None
    assert result["object_matching_accuracy"] is None


def test_benchmark_blocks_unverified_pair_and_ambiguous_identity_mapping():
    report, annotations, pair, mapping = metric_fixture()
    with pytest.raises(ValueError, match="blocked"):
        benchmark(
            report,
            annotations,
            dict(pair, pair_validity_status="RELATIONSHIP_ONLY"),
            mapping,
        )
    with pytest.raises(ValueError, match="one-to-one"):
        benchmark(
            report,
            annotations,
            pair,
            mapping.model_copy(
                update={"ground_truth_to_prediction": {"a": "pred-a", "b": "pred-a"}}
            ),
        )
    with pytest.raises(ValueError, match="unavailable prediction"):
        benchmark(
            report,
            annotations,
            pair,
            mapping.model_copy(update={"ground_truth_to_prediction": {"a": "absent"}}),
        )


def test_abstentions_and_nonexhaustive_inventory_do_not_invent_rates():
    report, annotations, pair, mapping = metric_fixture()
    report = copy.deepcopy(report)
    report["events"][0]["change"] = "UNKNOWN"
    result = benchmark(
        report,
        annotations.model_copy(update={"exhaustive_object_inventory": False}),
        pair,
        mapping,
    )
    assert result["abstentions"] == 1 and result["false_positive_rate"] is None
    assert (
        result["false_removal_rate"] is None
        and "physical_change_precision" not in result
    )


def test_annotation_api_is_separate_from_benchmark_readiness():
    with TestClient(app) as client:
        page = client.get("/annotate")
        assert page.status_code == 200 and "Playback is independent" in page.text
        response = client.post(
            "/annotations/validate",
            json={
                "scene_id": "unit",
                "changes": [
                    {
                        "object_id": "a",
                        "label": "unit",
                        "change_type": "UNKNOWN",
                        "annotation_source": "unit fixture",
                    }
                ],
            },
        )
        assert (
            response.status_code == 200 and response.json()["benchmark_ready"] is False
        )
        bad = client.post(
            "/annotations/validate",
            json={
                "scene_id": "unit",
                "changes": [
                    {
                        "object_id": "a",
                        "label": "unit",
                        "change_type": "UNKNOWN",
                        "annotation_source": "unit fixture",
                        "annotation_quality": "INDEPENDENTLY_REVIEWED",
                    }
                ],
            },
        )
        assert bad.status_code == 422
