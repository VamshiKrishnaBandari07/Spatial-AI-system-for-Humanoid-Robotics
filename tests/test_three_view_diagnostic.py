"""
test_three_view_diagnostic.py

Comprehensive test suite covering scale changes, rotated cameras, translated origins,
zero baseline, insufficient parallax, points behind cameras, malformed inputs, 
NaN/infinity, singular intrinsics, inconsistent A/B matches, and withheld-view failures.
"""

import numpy as np
import pytest
from spatialmemory.three_view_diagnostic import triangulate_and_validate_three_view

def get_base_setup():
    K = np.array([[500, 0, 320], [0, 500, 240], [0, 0, 1]], dtype=float)
    T_a = np.eye(4, dtype=float)
    T_b = np.eye(4, dtype=float); T_b[0, 3] = -0.5
    T_c = np.eye(4, dtype=float); T_c[0, 3] = -1.0
    pts_a = np.array([[320.0, 240.0]], dtype=float)
    pts_b = np.array([[195.0, 240.0]], dtype=float)
    pts_c = np.array([[70.0, 240.0]], dtype=float)
    direct_pts = np.array([[0.0, 0.0, 2.0]], dtype=float)
    return K, T_a, pts_a, T_b, pts_b, T_c, pts_c, direct_pts

def test_scale_changes():
    K, T_a, pts_a, T_b, pts_b, T_c, pts_c, _ = get_base_setup()
    for scale in [1e-7, 1.0, 1e8]:
        Ta_s = T_a.copy()
        Tb_s = T_b.copy(); Tb_s[0, 3] *= scale
        Tc_s = T_c.copy(); Tc_s[0, 3] *= scale
        direct_s = np.array([[0.0, 0.0, 2.0 * scale]], dtype=float)

        res = triangulate_and_validate_three_view(
            K, Ta_s, pts_a, K, Tb_s, pts_b, K, Tc_s, pts_c, direct_s,
            min_parallax_deg=0.1, max_reproj_error=2.0
        )
        assert res["final_valid_mask"][0]
        assert res["all_per_point_under_2px"]

def test_rotated_cameras_and_translated_origins():
    K, _, pts_a, _, pts_b, _, pts_c, _ = get_base_setup()
    # Rotate cameras slightly and translate origin in world space
    from scipy.spatial.transform import Rotation
    R_rot = Rotation.from_euler('y', 5, degrees=True).as_matrix()
    
    T_a = np.eye(4, dtype=float); T_a[:3, :3] = R_rot; T_a[:3, 3] = [10.0, 20.0, 30.0]
    T_b = np.eye(4, dtype=float); T_b[:3, :3] = R_rot; T_b[:3, 3] = [9.5, 20.0, 30.0]
    T_c = np.eye(4, dtype=float); T_c[:3, :3] = R_rot; T_c[:3, 3] = [9.0, 20.0, 30.0]

    # Re-project synthetic point [10.0, 20.0, 32.0] into rotated cameras
    # Local cam A point: R_rot.T @ ([10,20,32] - [10,20,30]) = R_rot.T @ [0, 0, 2]
    # Keep test execution robust by using valid synthetic projections:
    p_world = np.array([10.0, 20.0, 32.0], dtype=float)
    direct_pts = np.array([p_world], dtype=float)

    def proj(T, pt):
        cam = T[:3, :3] @ pt + T[:3, 3]
        pix = (K @ cam)[:2] / cam[2]
        return pix

    pa = proj(T_a, p_world).reshape(1, 2)
    pb = proj(T_b, p_world).reshape(1, 2)
    pc = proj(T_c, p_world).reshape(1, 2)

    res = triangulate_and_validate_three_view(
        K, T_a, pa, K, T_b, pb, K, T_c, pc, direct_pts,
        min_parallax_deg=0.1, max_reproj_error=2.0
    )
    assert res["final_valid_mask"][0]

def test_zero_baseline():
    K, T_a, pts_a, _, pts_b, _, pts_c, direct_pts = get_base_setup()
    with pytest.raises(ValueError, match="Zero or near-zero baseline"):
        triangulate_and_validate_three_view(
            K, T_a, pts_a, K, T_a, pts_b, K, T_a, pts_c, direct_pts
        )

def test_insufficient_parallax():
    K, T_a, pts_a, _, _, T_c, pts_c, direct_pts = get_base_setup()
    T_b_tiny = T_a.copy()
    T_b_tiny[0, 3] = -1e-5
    # Reproject the SAME world point after changing the camera baseline.
    camera_point = T_b_tiny[:3, :3] @ direct_pts[0] + T_b_tiny[:3, 3]
    projected = K @ camera_point
    pts_b = (projected[:2] / projected[2])[None, :]
    res = triangulate_and_validate_three_view(
        K, T_a, pts_a, K, T_b_tiny, pts_b, K, T_c, pts_c, direct_pts,
        min_parallax_deg=5.0,
    )
    assert not res["triangulation_valid_mask"][0]
    assert not res["final_valid_mask"][0]
    assert res["per_point_rejection_reasons"][0].startswith("Insufficient parallax")


def test_points_behind_cameras():
    K, T_a, pts_a, T_b, pts_b, T_c, pts_c, _ = get_base_setup()
    # Point behind camera (Z = -2.0)
    direct_pts = np.array([[0.0, 0.0, -2.0]], dtype=float)
    res = triangulate_and_validate_three_view(
        K, T_a, pts_a, K, T_b, pts_b, K, T_c, pts_c, direct_pts
    )
    assert np.isnan(res["mean_head_3d_divergence"])

def test_malformed_inputs_and_shapes():
    K, T_a, pts_a, T_b, pts_b, T_c, pts_c, direct_pts = get_base_setup()
    bad_pts = np.array([[320.0]], dtype=float) # shape (N, 1) instead of (N, 2)
    with pytest.raises(ValueError, match="must have shape"):
        triangulate_and_validate_three_view(
            K, T_a, bad_pts, K, T_b, pts_b, K, T_c, pts_c, direct_pts
        )

def test_nan_and_infinity():
    K, T_a, pts_a, T_b, pts_b, T_c, pts_c, direct_pts = get_base_setup()
    bad_pts = pts_a.copy()
    bad_pts[0, 0] = np.nan
    with pytest.raises(ValueError, match="non-finite values"):
        triangulate_and_validate_three_view(
            K, T_a, bad_pts, K, T_b, pts_b, K, T_c, pts_c, direct_pts
        )

def test_singular_intrinsics():
    K, T_a, pts_a, T_b, pts_b, T_c, pts_c, direct_pts = get_base_setup()
    singular_K = np.zeros((3, 3), dtype=float)
    with pytest.raises(ValueError, match="intrinsics are singular"):
        triangulate_and_validate_three_view(
            singular_K, T_a, pts_a, K, T_b, pts_b, K, T_c, pts_c, direct_pts
        )

def test_inconsistent_ab_matches():
    K, T_a, pts_a, T_b, pts_b, T_c, pts_c, direct_pts = get_base_setup()
    # Inconsistent match pushing reprojection error high
    bad_pts_b = pts_b.copy() + 50.0 
    res = triangulate_and_validate_three_view(
        K, T_a, pts_a, K, T_b, bad_pts_b, K, T_c, pts_c, direct_pts
    )
    assert not res["final_valid_mask"][0]
    assert res["per_point_rejection_reasons"][0].startswith("Reprojection error fails")


def test_withheld_view_failures():
    K, T_a, pts_a, T_b, pts_b, T_c, pts_c, direct_pts = get_base_setup()
    # Corrupt view C observations so withheld check fails < 2.0 px limit
    bad_pts_c = pts_c.copy() + 10.0
    res = triangulate_and_validate_three_view(
        K, T_a, pts_a, K, T_b, pts_b, K, T_c, bad_pts_c, direct_pts,
        max_reproj_error=2.0
    )
    assert not res["all_per_point_under_2px"]
    assert not res["final_valid_mask"][0]
