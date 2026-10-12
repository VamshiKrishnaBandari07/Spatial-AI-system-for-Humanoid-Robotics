import numpy as np
import pytest
from spatialmemory.object_geometry import triangulate_checked
from spatialmemory.geometry import project
from spatialmemory.pixel_mapping import native_to_geometry_matrix


def scene():
    k = np.array([[500.0, 0, 320], [0, 500.0, 240], [0, 0, 1.0]])
    a = np.eye(4)
    b = np.eye(4)
    b[0, 3] = -0.4
    points = np.array(
        [[-0.1, -0.1, 3], [0.1, -0.1, 3.1], [0.1, 0.1, 2.9], [-0.1, 0.1, 3.2]]
    )
    pa, _ = project(points, k, a)
    pb, _ = project(points, k, b)
    return points, pa, pb, k, a, b


def test_triangulation_recovers_known_world_and_transform_direction():
    points, pa, pb, k, a, b = scene()
    result = triangulate_checked(pa, pb, k, k, a, b, 1.0, 2.0)
    assert result.valid.all()
    np.testing.assert_allclose(result.points, points, atol=1e-8)
    assert result.reprojection_errors.max() < 1e-8


def test_no_parallax_cannot_create_object_geometry():
    _, pa, _, k, a, _ = scene()
    result = triangulate_checked(pa, pa, k, k, a, a, 1.0, 2.0)
    assert not result.valid.any()


def test_bad_correspondences_fail_reprojection_gates():
    _, pa, pb, k, a, b = scene()
    pb[:, 1] += 30
    result = triangulate_checked(pa, pb, k, k, a, b, 1.0, 2.0)
    assert not result.valid.any()


def test_behind_camera_points_are_rejected():
    _, _, _, k, a, b = scene()
    points = np.array([[0.0, 0.0, -3.0]])
    pa, _ = project(points, k, a)
    pb, _ = project(points, k, b)
    assert not triangulate_checked(pa, pb, k, k, a, b, 1.0, 2.0).valid.any()


def test_reflected_pose_and_invalid_intrinsics_are_rejected():
    _, pa, pb, k, a, b = scene()
    bad = b.copy()
    bad[0, 0] = -1
    with pytest.raises(ValueError, match="right-handed"):
        triangulate_checked(pa, pb, k, k, a, bad, 1.0, 2.0)
    badk = k.copy()
    badk[0, 0] = -500
    with pytest.raises(ValueError, match="intrinsics"):
        triangulate_checked(pa, pb, badk, k, a, b, 1.0, 2.0)


def test_native_mapping_recovers_projection_on_portrait_input():
    native = np.array([[700.0, 0, 232], [0, 710.0, 416], [0, 0, 1.0]])
    mapping = native_to_geometry_matrix((832, 464), (518, 518), "pad")
    modeled = mapping @ native
    np.testing.assert_allclose(np.linalg.inv(mapping) @ modeled, native, atol=1e-8)
    # Whole image corners stay within the padded model canvas.
    corners = mapping @ np.array([[0, 463], [0, 831], [1, 1]])
    assert corners[:2].min() > -1 and corners[:2].max() < 518


def test_geometry_supported_views_need_accepted_pairs_and_distinct_rays():
    from spatialmemory.contracts import Config
    from spatialmemory.memory import Observation
    from spatialmemory.object_geometry import geometric_view_support

    config = Config(geometry_supported_views_enabled=True)
    observations = [
        Observation(
            id=str(i),
            frame_id=str(i),
            timestamp=i,
            label="cup",
            descriptor=[1, 0],
            position=[0, 0, 2],
            viewpoint=str(i),
            provenance={"camera_center": [x, 0, 0]},
        )
        for i, x in enumerate([-0.1, 0.1])
    ]
    assert (
        geometric_view_support(observations, [0, 0, 2], config)["supported_views"] == 0
    )
    pair = {"before_id": "0", "after_id": "1", "accepted": True, "gate": None}
    for observation in observations:
        observation.provenance["multiview_geometry"] = {"candidate_pairs": [pair]}
    assert (
        geometric_view_support(observations, [0, 0, 2], config)["supported_views"] == 2
    )
    observations[1].provenance["camera_center"] = [-0.1, 0, 0]
    assert (
        geometric_view_support(observations, [0, 0, 2], config)["supported_views"] == 1
    )


def test_masked_flow_checks_native_correspondences_and_rejects_missing_texture():
    from spatialmemory.object_geometry import (
        ObjectFeatures,
        masked_flow_correspondences,
    )
    from spatialmemory.memory import Observation

    obs = Observation(id="a", frame_id="a", timestamp=0, label="cup", viewpoint="a")
    random = np.random.default_rng(9).integers(0, 255, (128, 128), dtype=np.uint8)
    mask = np.ones((128, 128), np.uint8) * 255

    def features(gray):
        return ObjectFeatures(
            obs, np.empty((0, 2)), None, np.eye(3), np.eye(4), gray, mask
        )

    a, b = masked_flow_correspondences(
        features(random), features(np.roll(random, 2, axis=1))
    )
    assert len(a) >= 4
    np.testing.assert_allclose(b - a, np.tile([2.0, 0.0], (len(a), 1)), atol=0.2)
    empty, _ = masked_flow_correspondences(
        features(np.zeros_like(random)), features(np.zeros_like(random))
    )
    assert len(empty) == 0


def test_crossview_free_space_needs_real_texture_and_depth_agreement():
    from types import SimpleNamespace
    from spatialmemory.visibility import crossview_free_surface

    k = np.array([[60.0, 0, 32], [0, 60.0, 32], [0, 0, 1.0]])
    a = np.eye(4)
    b = np.eye(4)
    b[0, 3] = -0.3
    texture = np.random.default_rng(8).integers(0, 255, (65, 65, 3), dtype=np.uint8)
    surfaces = {
        name: {"depth": np.full((65, 65), 3.0), "intrinsics": k} for name in ["a", "b"]
    }
    reconstruction = SimpleNamespace(
        surfaces=surfaces,
        poses={"a": a, "b": b},
        features={
            "a": (None, None, texture),
            "b": (None, None, np.roll(texture, -6, axis=1)),
        },
        diagnostics={},
    )
    result = crossview_free_surface([[0.0, 0.0, 2.0]], "a", reconstruction, 0.05)
    assert result["supported_fraction"] == 1 and result["supporting_frames"] == ["b"]
    surfaces["b"]["depth"][:] = 2.5
    assert (
        crossview_free_surface([[0.0, 0.0, 2.0]], "a", reconstruction, 0.05)[
            "supported_fraction"
        ]
        == 0
    )
    surfaces["b"]["depth"][:] = 3.0
    reconstruction.features = {
        name: (None, None, np.zeros_like(texture)) for name in ["a", "b"]
    }
    assert (
        crossview_free_surface([[0.0, 0.0, 2.0]], "a", reconstruction, 0.05)[
            "supported_fraction"
        ]
        == 0
    )
