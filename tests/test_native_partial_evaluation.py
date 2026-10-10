"""Synthetic contract fixtures for partial evaluation; not real accuracy evidence."""

import pytest
from scripts.review_3rscan import evaluate


def fixture():
    review = {
        "scene_id": "synthetic-evaluation-contract",
        "native_split": "train",
        "unsupported_categories": ["ADDED", "REMOVED", "UNCHANGED"],
        "objects": [
            {
                "object_id": "positive",
                "label": "trash can",
                "change_type": "MOVED",
                "before_position": [0, 0, 0],
                "after_position": [0.1, 0, 0],
            },
            {
                "object_id": "unsupported",
                "label": "chair",
                "change_type": "UNKNOWN",
                "before_position": None,
                "after_position": None,
            },
        ],
    }
    report = {
        "status": "complete",
        "registration": {"valid": True},
        "scene_id": review["scene_id"],
        "events": [],
    }
    return review, report


def test_partial_evaluation_preserves_missing_predictions_and_unscored_unknowns():
    review, report = fixture()
    result = evaluate(review, report)
    assert result["sample_count"] == 1 and result["unscored_native_objects"] == 1
    assert result["confusion_matrix"]["MOVED"]["UNKNOWN"] == 1
    assert result["per_class"]["MOVED"] == {
        "precision": None,
        "recall": 0.0,
        "f1": 0.0,
        "support": 1,
    }
    assert (
        result["full_detection_precision"] is None
        and result["object_matching_accuracy"] is None
    )


def test_partial_evaluation_rejects_bad_registration_and_mismatched_scene():
    review, report = fixture()
    report["registration"]["valid"] = False
    with pytest.raises(ValueError, match="gated inference"):
        evaluate(review, report)
    report["registration"]["valid"] = True
    report["scene_id"] = "different-scene"
    with pytest.raises(ValueError, match="scene mismatch"):
        evaluate(review, report)


def test_competing_candidate_predictions_cannot_become_a_verified_positive():
    review, report = fixture()
    report["events"] = [
        {
            "id": str(i),
            "object_id": str(i),
            "semantic_label": "trash",
            "change": "MOVED",
            "before_position": [0.01 * i, 0, 0],
            "after_position": None,
        }
        for i in range(2)
    ]
    result = evaluate(review, report)
    assert result["objects"][0]["prediction"] == "UNKNOWN"
    assert result["per_class"]["MOVED"]["recall"] == 0
    report["events"] = report["events"][:1]
    assert evaluate(review, report)["per_class"]["MOVED"]["recall"] == 1
