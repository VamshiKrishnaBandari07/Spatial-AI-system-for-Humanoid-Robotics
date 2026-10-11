import numpy as np
import pytest

from spatialmemory.semantics import localized_surface_points
from spatialmemory.semantics import visibility_evidence
from spatialmemory.contracts import Config
from types import SimpleNamespace


def surface_and_mask():
    y, x = np.indices((3, 3))
    points = np.stack([x, y, np.ones_like(x)], axis=-1).astype(float)
    mask = np.zeros((3, 3), bool)
    mask[2] = True
    confidence = np.full((3, 3), 100.0)
    confidence[mask] = 1
    return {
        "points": points,
        "confidence": confidence,
        "depth": np.ones((3, 3)),
        "intrinsics": np.eye(3),
        "depth_confidence": np.full((3, 3), 10.0),
    }, mask


def test_padding_is_never_localization_support():
    surface, mask = surface_and_mask()
    surface["valid_pixels"] = np.zeros(mask.shape, bool)
    surface["valid_pixels"][0] = True
    points, audit = localized_surface_points(mask, surface, np.eye(4), 0.08)
    assert len(points) == 0
    assert audit["point_head_supported_pixels"] == 0
    assert audit["additional_depth_head_agreement_pixels"] == 0


def test_depth_head_agreement_localizes_low_point_confidence_region():
    surface, mask = surface_and_mask()
    points, audit = localized_surface_points(mask, surface, np.eye(4), 0.08)
    np.testing.assert_allclose(points, surface["points"][mask])
    assert audit["point_head_supported_pixels"] == 0
    assert audit["additional_depth_head_agreement_pixels"] == 3
    assert not audit["independent_geometry_validation"]


def test_depth_head_disagreement_preserves_missing_localization():
    surface, mask = surface_and_mask()
    surface["points"] += 1
    points, audit = localized_surface_points(mask, surface, np.eye(4), 0.08)
    assert len(points) == 0
    assert audit["additional_depth_head_agreement_pixels"] == 0


def test_unavailable_depth_confidence_does_not_promote_points():
    surface, mask = surface_and_mask()
    surface.pop("depth_confidence")
    points, audit = localized_surface_points(mask, surface, np.eye(4), 0.08)
    assert len(points) == 0
    assert not audit["depth_head_available"]


def test_camera_world_transform_applied_to_depth_points():
    surface, mask = surface_and_mask()
    pose = np.eye(4)
    pose[:3, :3] = [[0, -1, 0], [1, 0, 0], [0, 0, 1]]
    pose[:3, 3] = [0.2, 0.4, 0.5]
    surface["points"] = (surface["points"] - pose[:3, 3]) @ pose[:3, :3]
    points, _ = localized_surface_points(mask, surface, pose, 0.001)
    np.testing.assert_allclose(points, surface["points"][mask])


def test_invalid_depth_and_low_depth_confidence_do_not_promote_points():
    surface, mask = surface_and_mask()
    surface["depth"][mask] = np.nan
    assert len(localized_surface_points(mask, surface, np.eye(4), 0.08)[0]) == 0
    surface, mask = surface_and_mask()
    surface["depth_confidence"][mask] = 0
    assert len(localized_surface_points(mask, surface, np.eye(4), 0.08)[0]) == 0


def test_misaligned_depth_dimensions_rejected():
    surface, mask = surface_and_mask()
    surface["depth_confidence"] = np.ones((2, 2))
    with pytest.raises(ValueError, match="dimensions"):
        localized_surface_points(mask, surface, np.eye(4), 0.08)


@pytest.mark.parametrize("supported", [True, False])
def test_visibility_uses_same_geometry_agreement_gate(supported):
    surface, _ = surface_and_mask()
    if not supported:
        surface["points"] += 1
    scene = SimpleNamespace(surfaces={"frame": surface}, poses={"frame": np.eye(4)})
    obj = SimpleNamespace(position=[0, 1, 0.5], extent=None)
    result = visibility_evidence(obj, scene, Config(voxel_size=0.05))
    assert result["views"][0]["states"] == ["FREE" if supported else "UNKNOWN"]
    assert not result["metric"]
