from types import SimpleNamespace

import numpy as np
import pytest

from spatialmemory.association import unit_descriptor, match_objects
from spatialmemory.contracts import Config
from spatialmemory import registration


def test_large_finite_descriptor_preserves_identity_direction():
    large = unit_descriptor([1e308, 1e308])
    np.testing.assert_allclose(large, unit_descriptor([1, 1]))
    assert np.isfinite(large).all()
    assert np.linalg.norm(large) == pytest.approx(1)


@pytest.mark.parametrize("descriptor", [[0, 0], [1e-20, 0], [np.inf, 1]])
def test_invalid_descriptor_remains_rejected(descriptor):
    with pytest.raises(ValueError):
        unit_descriptor(descriptor)


@pytest.mark.parametrize(
    "source,target",
    [
        (np.zeros((6, 2)), np.zeros((6, 2))),
        (np.zeros((6, 3)), np.zeros((5, 3))),
        (np.full((6, 3), np.nan), np.zeros((6, 3))),
    ],
)
def test_invalid_registration_inputs_fail_clearly(source, target):
    with pytest.raises(ValueError, match="finite Nx3"):
        registration.ransac(source, target)


@pytest.mark.parametrize(
    "count,valid,heldout", [(9, False, 0), (16, True, 4), (25, True, 5)]
)
def test_phone_registration_requires_heldout_support(
    monkeypatch, count, valid, heldout
):
    points = np.random.default_rng(7).normal(size=(count, 3))
    reconstruction = SimpleNamespace(
        points=points,
        features={"frame": (np.zeros((count, 2)), np.arange(count), None)},
        tracks={("frame", i): i for i in range(count)},
    )
    monkeypatch.setattr(
        registration, "match", lambda a, b: [(i, i) for i in range(count)]
    )
    result = registration.register(
        reconstruction, reconstruction, Config(registration_min_heldout=4)
    )
    assert result.valid is valid
    assert result.heldout_count == heldout
    if not valid:
        assert "Insufficient held-out" in result.reason


@pytest.mark.parametrize(
    "field", ["sample_seconds", "focal_px", "registration_threshold", "voxel_size"]
)
def test_nonfinite_configuration_rejected(field):
    with pytest.raises(ValueError):
        Config(**{field: float("inf")})


def test_reserved_phone_holdout_still_rejects_bad_alignment(monkeypatch):
    count = 16
    source = np.random.default_rng(19).normal(size=(count, 3))
    target = source.copy()
    heldout = np.random.default_rng(7).permutation(count)[:4]
    target[heldout] += 100

    def scene(points):
        return SimpleNamespace(
            points=points,
            features={"frame": (np.zeros((count, 2)), np.arange(count), None)},
            tracks={("frame", i): i for i in range(count)},
        )

    monkeypatch.setattr(
        registration, "match", lambda a, b: [(i, i) for i in range(count)]
    )
    result = registration.register(
        scene(target), scene(source), Config(registration_min_heldout=4)
    )
    assert not result.valid
    assert result.heldout_count == 4
    assert result.heldout_inliers == 0
    assert result.reason == "Held-out registration validation failed"


@pytest.mark.parametrize("position", [[1, 2], [[1], [2, 3]], ["invalid", 0, 0]])
def test_invalid_localization_diagnostics_do_not_crash(position):
    a = SimpleNamespace(
        id="a",
        label="chair",
        descriptor=[1, 0],
        position=position,
        extent=None,
        observations=[],
        distinct_views=2,
    )
    b = SimpleNamespace(
        id="b",
        label="chair",
        descriptor=[1, 0],
        position=[0, 0, 0],
        extent=None,
        observations=[],
        distinct_views=2,
    )
    audit = []
    assert (
        match_objects([a], [b], Config(association_mode="multiview"), diagnostics=audit)
        == []
    )
    assert audit[0]["gate"] == "invalid_localization"
    assert audit[0]["registered_centroid_distance"] is None
