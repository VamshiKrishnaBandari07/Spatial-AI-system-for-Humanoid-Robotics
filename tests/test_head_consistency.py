"""head_consistency must recover known, injected camera-head errors (synthetic only)."""

import numpy as np
import pytest

from spatialmemory.head_consistency import head_consistency, interpret
from spatialmemory.pixel_mapping import native_to_geometry_matrix

H = W = 518
K_TRUE = np.array([[420.0, 0, 259.0], [0, 420.0, 259.0], [0, 0, 1.0]])
POSE = np.hstack([np.eye(3), np.zeros((3, 1))])  # camera at origin; world == camera
MAPPING = native_to_geometry_matrix((1920, 1080), (H, W), "pad")  # sx ~ 0.27


def scene(seed=0):
    rng = np.random.default_rng(seed)
    depth = rng.uniform(1.5, 4.0, (H, W))
    yy, xx = np.indices((H, W))
    rays = np.stack([xx, yy, np.ones_like(xx)], -1) @ np.linalg.inv(K_TRUE).T
    return rays * depth[..., None], rng.uniform(0.1, 1.0, (H, W))


def run(k_used):
    points, conf = scene()
    return head_consistency(points, conf, POSE, k_used, MAPPING)


def test_consistent_heads_pass_with_negligible_residual():
    view = run(K_TRUE)
    assert view["residual_model_px"]["p99"] < 1e-6 and view["fraction_within_native_limit"] == 1.0
    assert interpret(view) == "passes the gate"


def test_constant_principal_point_error_is_recovered_as_shift():
    k = K_TRUE.copy(); k[0, 2] += 3.0; k[1, 2] -= 2.0
    view = run(k)
    assert view["fit"]["shift_x_model_px"] == pytest.approx(3.0, abs=0.05)
    assert view["fit"]["shift_y_model_px"] == pytest.approx(-2.0, abs=0.05)
    assert abs(view["fit"]["relative_focal_error_x"]) < 1e-3
    assert view["fit"]["fraction_of_variance_explained_by_shift_and_focal"] > 0.95
    assert "constant shift" in interpret(view)
    # ~3.6 model px is ~13 native px for this frame: far beyond the unchanged 2 px limit.
    assert view["fraction_within_native_limit"] == 0.0
    assert view["residual_native_px"]["median"] == pytest.approx(3.6 / MAPPING[0, 0], rel=0.02)


def test_focal_disagreement_is_recovered_as_radial_scale():
    k = K_TRUE.copy(); k[0, 0] *= 1.03; k[1, 1] *= 1.03
    view = run(k)
    assert view["fit"]["relative_focal_error_x"] == pytest.approx(0.03, abs=1e-3)
    assert view["fit"]["relative_focal_error_y"] == pytest.approx(0.03, abs=1e-3)
    assert abs(view["fit"]["shift_x_model_px"]) < 0.05
    assert "radial scale" in interpret(view)


def test_unstructured_noise_is_not_mistaken_for_a_systematic_error():
    points, conf = scene()
    noise = np.random.default_rng(1).normal(0, 0.02, points.shape)  # head disagreement
    view = head_consistency(points + noise, conf, POSE, K_TRUE, MAPPING)
    assert view["fit"]["fraction_of_variance_explained_by_shift_and_focal"] < 0.3
    assert "unstructured" in interpret(view)


def test_gate_limit_is_expressed_unchanged_in_model_pixels():
    view = run(K_TRUE)
    assert view["native_limit_px"] == 2.0  # never relaxed by the diagnostic
    assert view["native_limit_in_model_px"] == pytest.approx(0.54, abs=0.01)


def test_behind_camera_or_nonfinite_pixels_are_excluded_not_counted():
    points, conf = scene()
    points[:50] = np.nan
    points[50:100, :, 2] *= -1
    view = head_consistency(points, conf, POSE, K_TRUE, MAPPING)
    assert view["pixels_evaluated"] < H * W * 0.5 * 0.9  # median-confidence filter halves it
    assert view["residual_model_px"]["p99"] < 1e-6
    with pytest.raises(ValueError):
        head_consistency(np.full((H, W, 3), np.nan), conf, POSE, K_TRUE, MAPPING)


def _write_run(root, shifted_role):
    import json
    import cv2
    from spatialmemory.pixel_mapping import native_to_geometry_matrix  # noqa: F401

    config = {"neural_preprocessing": "pad", "neural_frames_per_video": 3,
              "triangulation_max_reprojection_px": 2.0}
    (root / "report.json").write_text(json.dumps({"config": config}))
    for role in ("before", "after"):
        (root / role).mkdir()
        (root / "neural" / role).mkdir(parents=True)
        frames = []
        for i in range(3):
            path = root / role / f"frame_{i}.png"
            cv2.imwrite(str(path), np.zeros((1920, 1080, 3), np.uint8))
            frames.append({"id": f"{role}_{i}", "index": i, "timestamp": float(i), "path": str(path),
                           "sharpness": 100.0 + i, "exposure": 0.5, "features": 500 + 10 * i,
                           "duplicate_distance": None})
        (root / role / "manifest.json").write_text(json.dumps({"frames": frames}))
        k = K_TRUE.copy()
        if role == shifted_role:
            k[0, 2] += 3.0  # camera head disagrees with the point head
        points, conf = scene()
        np.savez(root / "neural" / role / "predictions.npz",
                 world_points=np.repeat(points[None, None], 3, 1), world_points_conf=np.repeat(conf[None, None], 3, 1),
                 extrinsics=np.repeat(POSE[None, None], 3, 1), intrinsics=np.repeat(k[None, None], 3, 1))


def test_analyse_run_reads_a_saved_run_and_separates_roles(tmp_path):
    from spatialmemory.head_consistency import analyse_run

    _write_run(tmp_path, shifted_role="after")
    before_snapshot = sorted(p.name for p in tmp_path.rglob("*"))
    result = analyse_run(tmp_path)
    by_role = {}
    for view in result["views"]:
        by_role.setdefault(view["role"], []).append(view)
    assert {r: len(v) for r, v in by_role.items()} == {"before": 3, "after": 3}
    assert all(v["fraction_within_native_limit"] == 1.0 for v in by_role["before"])
    assert all(v["fraction_within_native_limit"] == 0.0 and "constant shift" in v["reading"]
               for v in by_role["after"])
    assert by_role["after"][0]["native_shape"] == [1920, 1080]
    assert sorted(p.name for p in tmp_path.rglob("*")) == before_snapshot, "run directory must stay read-only"


def test_anisotropic_resize_uses_both_axes_at_strict_boundary():
    points, confidence = scene()
    mapping = native_to_geometry_matrix((832, 464), (H, W), 'pad')
    intrinsic = K_TRUE.copy()
    intrinsic[1, 2] += 1.25
    result = head_consistency(points, confidence, POSE, intrinsic, mapping)
    assert result['residual_native_px']['median'] == pytest.approx(1.25 / mapping[1, 1])
    assert result['fraction_within_native_limit'] == 0.0
