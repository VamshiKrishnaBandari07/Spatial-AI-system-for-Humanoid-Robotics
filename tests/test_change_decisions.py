"""classify_object/compare must fail closed on invalid or non-boolean evidence."""

import numpy as np
import pytest

from spatialmemory.change import classify_object
from spatialmemory.contracts import Change

NAN, INF = float("nan"), float("inf")
B, A = [0.0, 0.0, 1.0], [0.01, 0.0, 1.0]


@pytest.mark.parametrize("before,after", [
    ([NAN, 0, 1], A), (B, [0, NAN, 1]), (B, [INF, 0, 1]), ([-INF, 0, 1], A),
])
def test_unusable_positions_are_unknown_never_unchanged_or_moved(before, after):
    assert classify_object(before, after, True, False, 0.1, 0.02) == Change.UNKNOWN


def test_unusable_position_is_not_treated_as_absence():
    # A NaN position is an unusable observation, not "object absent": even with
    # positive free-space evidence it must not become REMOVED/ADDED.
    assert classify_object([NAN, 0, 1], None, True, True, 0.1, 0.02) == Change.UNKNOWN
    assert classify_object(None, [0, NAN, 1], True, True, 0.1, 0.02) == Change.UNKNOWN


@pytest.mark.parametrize("evidence", [
    {"supported_free_views": 0, "free": False}, "false", "yes", 1, 2.0, [True], np.array([True]), None,
])
def test_non_boolean_evidence_never_promotes_added_or_removed(evidence):
    assert classify_object(B, None, True, evidence, 0.1, 0.02) == Change.UNKNOWN
    assert classify_object(None, A, True, evidence, 0.1, 0.02) == Change.UNKNOWN


def test_numpy_bool_true_is_accepted_as_positive_evidence():
    assert classify_object(B, None, True, np.True_, 0.1, 0.02) == Change.REMOVED
    assert classify_object(None, A, True, np.True_, 0.1, 0.02) == Change.ADDED


@pytest.mark.parametrize("identity", [1, "true", {"verified": False}, None])
def test_identity_must_be_literal_true_for_moved_or_unchanged(identity):
    assert classify_object(B, [1, 0, 1], identity, False, 0.1, 0.02) == Change.UNKNOWN


def test_nothing_observed_anywhere_cannot_be_added_or_removed():
    assert classify_object(None, None, True, True, 0.1, 0.02) == Change.UNKNOWN


@pytest.mark.parametrize("threshold,uncertainty", [
    (NAN, 0.02), (0.1, NAN), (0.0, 0.02), (-0.1, 0.02), (0.1, -1.0), (INF, 0.02),
])
def test_invalid_configuration_raises_instead_of_deciding(threshold, uncertainty):
    with pytest.raises(ValueError, match="threshold must be finite"):
        classify_object(B, A, True, False, threshold, uncertainty)


def test_displacement_within_uncertainty_of_threshold_remains_unknown():
    assert classify_object(B, [0.1, 0, 1], True, False, 0.1, 0.02) == Change.UNKNOWN


def _scene():
    from spatialmemory.contracts import Config
    from spatialmemory.registration import ransac

    rng = np.random.default_rng(3)
    base = rng.normal(size=(60, 3))
    registration, _ = ransac(base, base, 0.01)
    config = Config()
    blob = np.array([6.0, 6.0, 6.0]) + rng.normal(scale=config.voxel_size, size=(config.min_cluster + 20, 3))
    return np.vstack([base, blob]), base, registration, config


def test_compare_promotes_removed_only_on_literal_true_free_space_evidence():
    from spatialmemory.change import compare

    before, after, registration, config = _scene()
    promoted = compare(before, after, registration, config, free_after=lambda g: True)
    assert [e.change for e in promoted] == [Change.REMOVED]
    for evidence in ({"free": False}, "no", 1, None, np.array([True])):
        events = compare(before, after, registration, config, free_after=lambda g, e=evidence: e)
        assert [e.change for e in events] == [Change.UNKNOWN], evidence
        assert events[0].evidence["positive_free_space_evidence"] is False


def test_compare_without_free_space_callback_never_promotes():
    from spatialmemory.change import compare

    before, after, registration, config = _scene()
    assert [e.change for e in compare(before, after, registration, config)] == [Change.UNKNOWN]

@pytest.mark.parametrize('position', [[], [1, 2], [[1, 2, 3]], 'invalid', [1, 2, 3, 4]])
def test_malformed_position_cannot_establish_change(position):
    from spatialmemory.change import classify_object
    from spatialmemory.contracts import Change
    assert classify_object(position, None, True, True, .25, .01) == Change.UNKNOWN
    assert classify_object(None, position, True, True, .25, .01) == Change.UNKNOWN
