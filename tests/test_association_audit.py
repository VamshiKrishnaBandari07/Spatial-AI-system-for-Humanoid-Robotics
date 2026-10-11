from types import SimpleNamespace
import numpy as np
import pytest
from spatialmemory.association import match_objects
from spatialmemory.contracts import Config


def object_state(identifier, label="chair", descriptor=None, views=2):
    return SimpleNamespace(
        id=identifier,
        label=label,
        descriptor=descriptor or [1, 0, 0],
        position=[0, 0, 1],
        extent=[[0, 0, 1], [0.2, 0.2, 1.2]],
        distinct_views=views,
        position_spread=0.01,
        observations=[],
    )


@pytest.mark.parametrize(
    "label,descriptor,views,gate",
    [
        ("table", [1, 0, 0], 2, "semantic_incompatible"),
        ("chair", [0, 1, 0], 2, "appearance_below_floor"),
        ("chair", [1, 0, 0], 1, "insufficient_independent_views"),
    ],
)
def test_rejected_pairs_have_exact_gate_and_observed_metrics(
    label, descriptor, views, gate
):
    a, b = object_state("a"), object_state("b", label, descriptor, views)
    rows = []
    assert (
        match_objects([a], [b], Config(association_mode="multiview"), diagnostics=rows)
        == []
    )
    assert rows[0]["gate"] == gate
    assert rows[0]["appearance_cosine"] is not None
    assert rows[0]["registered_centroid_distance"] == 0
    assert rows[0]["calibrated"] is False
    assert np.allclose(rows[0]["descriptor_norms"], [1, 1])


def test_diagnostics_preserve_actual_assignment_and_ambiguity():
    config = Config(association_mode="multiview")
    a, b = [object_state("a")], [object_state("b")]
    rows = []
    assert match_objects(a, b, config) == match_objects(a, b, config, diagnostics=rows)
    assert rows[0]["assignment_status"] == "accepted" and rows[0]["gate"] is None
    rows = []
    result = match_objects(
        a, [object_state("b"), object_state("c")], config, diagnostics=rows
    )
    assert not result[0]["identity_verified"]
    assert any(row["gate"] == "assignment_ambiguity" for row in rows)


def test_unknown_events_explain_visibility_and_identity_gates():
    from spatialmemory.semantics import consensus
    from spatialmemory.contracts import Registration

    a, b = object_state("a", views=1), object_state("b", views=1)
    reconstruction = SimpleNamespace(surfaces={}, poses={})
    registration = Registration(
        valid=True,
        scale=1,
        rotation=np.eye(3).tolist(),
        translation=[0, 0, 0],
        inliers=10,
        correspondences=10,
        rmse=0,
        threshold=0.08,
        reason="synthetic fixture",
    )
    events, matches = consensus(
        [a],
        [b],
        reconstruction,
        reconstruction,
        registration,
        Config(association_mode="multiview", movement_mode="bounded"),
    )
    assert not matches and len(events) == 2
    for event in events:
        assert event.change.value == "UNKNOWN"
        assert "insufficient_independent_views" in event.evidence["decision_gates"]
        assert "unresolved_cross_recording_identity" in event.evidence["decision_gates"]
        assert "insufficient_positive_free_space" in event.evidence["decision_gates"]
        assert "Decision blocked by:" in event.explanation
