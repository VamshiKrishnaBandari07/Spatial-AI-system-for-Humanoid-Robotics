import pytest

from spatialmemory.object_state import ValidationFailureError, evaluate_object_state


def evidence(**updates):
    return dict(
        object_id="cup-1",
        object_name="cup",
        geometry_verified=True,
        wholly_outside_frustum=False,
        fully_occluded=False,
        identity_verified=False,
        position_verified=False,
        free_views=0,
        prior_object_verified=True,
        **updates,
    )


@pytest.mark.parametrize(
    "updates,state",
    [
        ({"wholly_outside_frustum": True}, "OUT_OF_VIEW"),
        ({"fully_occluded": True}, "HIDDEN"),
        ({"identity_verified": True, "position_verified": True}, "PRESENT"),
        ({"free_views": 2}, "REMOVED"),
    ],
)
def test_supported_states(updates, state):
    args = evidence()
    args.update(updates)
    result = evaluate_object_state(**args)
    assert result["state"] == state
    assert result["calibrated_confidence"] is None


@pytest.mark.parametrize(
    "updates",
    [
        {},
        {"free_views": 1},
        {"free_views": 2, "unresolved_identity": True},
        {"free_views": 2, "prior_object_verified": False},
        {"geometry_verified": False, "wholly_outside_frustum": True},
        {"fully_occluded": True, "free_views": 2},
        {
            "wholly_outside_frustum": True,
            "identity_verified": True,
            "position_verified": True,
        },
        {"identity_verified": True},
        {"free_views": True},
        {"fully_occluded": None},
    ],
)
def test_missing_or_conflicting_evidence_fails(updates):
    args = evidence()
    args.update(updates)
    with pytest.raises(ValidationFailureError):
        evaluate_object_state(**args)
