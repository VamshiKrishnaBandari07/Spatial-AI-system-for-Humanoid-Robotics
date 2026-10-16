"""
three_view_diagnostic.py

Standalone diagnostic module for multi-view geometry validation and VGGT head disagreement.
CRITICAL DOCUMENTATION REQUIREMENT:
    Intrinsics (K) and observation pixel coordinates (pts_a, pts_b, pts_c) MUST 
    use the exact same native-pixel coordinate system (including matching scaling, 
    resizing, and offset conventions).
"""

import numpy as np

def validate_camera_params(K: np.ndarray, T_cw: np.ndarray, name: str):
    """Validates finite, invertible pinhole intrinsics and finite rigid world-to-camera transforms."""
    if not isinstance(K, np.ndarray) or K.shape != (3, 3):
        raise ValueError(f"{name} intrinsics must be a 3x3 ndarray, got {type(K)} shape {getattr(K, 'shape', None)}")
    if not np.isfinite(K).all():
        raise ValueError(f"{name} intrinsics contain non-finite values (NaN or inf).")
    
    # Check singularity / invertibility via determinant
    det_k = np.linalg.det(K)
    if np.abs(det_k) < 1e-12:
        raise ValueError(f"{name} intrinsics are singular (det K = 0).")
    if K[0, 0] <= 0 or K[1, 1] <= 0:
        raise ValueError(f"{name} focal lengths must be positive.")

    if not isinstance(T_cw, np.ndarray) or T_cw.shape != (4, 4):
        raise ValueError(f"{name} extrinsics must be a 4x4 ndarray, got {type(T_cw)} shape {getattr(T_cw, 'shape', None)}")
    if not np.isfinite(T_cw).all():
        raise ValueError(f"{name} extrinsics contain non-finite values.")

    # Validate homogeneous bottom row
    expected_bottom = np.array([0.0, 0.0, 0.0, 1.0])
    if not np.allclose(T_cw[3, :], expected_bottom, atol=1e-6):
        raise ValueError(f"{name} extrinsics have an invalid homogeneous bottom row: {T_cw[3, :]}")

    # Validate rotation orthogonality and determinant
    R = T_cw[:3, :3]
    if not np.allclose(R.T @ R, np.eye(3), atol=1e-3) or not np.allclose(np.linalg.det(R), 1.0, atol=1e-3):
        raise ValueError(f"{name} rotation matrix is invalid (not orthogonal or det != 1).")

def validate_observations(arr: np.ndarray, expected_shape_col: int, name: str):
    """Validates exact observation shapes (N, expected_shape_col), finiteness, and non-emptiness."""
    if not isinstance(arr, np.ndarray):
        raise ValueError(f"{name} must be an ndarray, got {type(arr)}")
    if not np.isfinite(arr).all():
        raise ValueError(f"{name} contains non-finite values (NaN or inf).")
    if arr.ndim != 2 or arr.shape[1] != expected_shape_col:
        raise ValueError(f"{name} must have shape (N, {expected_shape_col}), got {arr.shape}")
    if arr.shape[0] == 0:
        raise ValueError(f"{name} cannot be empty (N=0).")

def triangulate_and_validate_three_view(
    K_a: np.ndarray, T_cw_a: np.ndarray, pts_a: np.ndarray,
    K_b: np.ndarray, T_cw_b: np.ndarray, pts_b: np.ndarray,
    K_c: np.ndarray, T_cw_c: np.ndarray, pts_c: np.ndarray,
    pts_3d_direct: np.ndarray,
    min_parallax_deg: float = 2.0,
    max_reproj_error: float = 2.0
) -> dict:
    """
    Scale-invariant three-view diagnostic comparing triangulated points against 
    direct VGGT head points. Keeps uncertainty explicitly unavailable.
    """
    # 1. Threshold & Parameter Sanity Checks
    if not np.isfinite(max_reproj_error) or max_reproj_error <= 0:
        raise ValueError(f"Invalid reprojection threshold: {max_reproj_error}. Must be finite and positive.")
    if not np.isfinite(min_parallax_deg) or min_parallax_deg < 0:
        raise ValueError(f"Invalid parallax threshold: {min_parallax_deg}.")

    validate_camera_params(K_a, T_cw_a, "Camera A")
    validate_camera_params(K_b, T_cw_b, "Camera B")
    validate_camera_params(K_c, T_cw_c, "Camera C")

    validate_observations(pts_a, 2, "pts_a")
    validate_observations(pts_b, 2, "pts_b")
    validate_observations(pts_c, 2, "pts_c")
    validate_observations(pts_3d_direct, 3, "pts_3d_direct")

    n_pts = pts_a.shape[0]
    if len(pts_b) != n_pts or len(pts_c) != n_pts or len(pts_3d_direct) != n_pts:
        raise ValueError("Point count mismatch across observations and direct-head points.")

    # 2. Camera centers and baseline calculation
    R_a, t_a = T_cw_a[:3, :3], T_cw_a[:3, 3]
    R_b, t_b = T_cw_b[:3, :3], T_cw_b[:3, 3]
    R_c, t_c = T_cw_c[:3, :3], T_cw_c[:3, 3]

    center_a = -R_a.T @ t_a
    center_b = -R_b.T @ t_b
    baseline = np.linalg.norm(center_b - center_a)
    if baseline < 1e-12:
        raise ValueError("Zero or near-zero baseline between camera A and B (degenerate camera setup).")

    # 3. Normalized extrinsics for scale-invariant DLT & Parallax
    def make_normalized_extrinsics(R, t, ca, b):
        t_prime = (R @ ca + t) / b
        T_prime = np.eye(4)
        T_prime[:3, :3] = R
        T_prime[:3, 3] = t_prime
        return T_prime

    T_cw_a_norm = make_normalized_extrinsics(R_a, t_a, center_a, baseline)
    T_cw_b_norm = make_normalized_extrinsics(R_b, t_b, center_a, baseline)

    P_a_norm = K_a @ T_cw_a_norm[:3, :]
    P_b_norm = K_b @ T_cw_b_norm[:3, :]

    center_a_norm = -R_a.T @ T_cw_a_norm[:3, 3]
    center_b_norm = -R_b.T @ T_cw_b_norm[:3, 3]

    triangulated_pts = np.full((n_pts, 3), np.nan)
    triangulation_valid_mask = np.zeros(n_pts, dtype=bool)
    rejection_reasons = []

    for i in range(n_pts):
        u1, v1 = pts_a[i]
        u2, v2 = pts_b[i]

        A = np.array([
            u1 * P_a_norm[2] - P_a_norm[0],
            v1 * P_a_norm[2] - P_a_norm[1],
            u2 * P_b_norm[2] - P_b_norm[0],
            v2 * P_b_norm[2] - P_b_norm[1]
        ])

        _, _, Vt = np.linalg.svd(A)
        X = Vt[-1]
        if np.abs(X[3]) < 1e-7:
            rejection_reasons.append("Singular DLT matrix / zero homogeneous coordinate")
            continue
        X_prime = X[:3] / X[3]

        ray_a_norm = X_prime - center_a_norm
        ray_b_norm = X_prime - center_b_norm
        norm_a, norm_b = np.linalg.norm(ray_a_norm), np.linalg.norm(ray_b_norm)
        if norm_a < 1e-6 or norm_b < 1e-6:
            rejection_reasons.append("Degenerate ray length in normalized frame")
            continue
        cos_angle = np.clip(np.dot(ray_a_norm, ray_b_norm) / (norm_a * norm_b), -1.0, 1.0)
        angle_deg = np.degrees(np.arccos(cos_angle))
        if angle_deg < min_parallax_deg:
            rejection_reasons.append(f"Insufficient parallax: {angle_deg:.2f} deg < {min_parallax_deg}")
            continue

        X_3d = baseline * X_prime + center_a

        cam_a = R_a @ X_3d + t_a
        cam_b = R_b @ X_3d + t_b
        cam_c = R_c @ X_3d + t_c

        if cam_a[2] <= 0 or cam_b[2] <= 0 or cam_c[2] <= 0:
            rejection_reasons.append("Cheirality violation (negative Z depth)")
            continue

        triangulated_pts[i] = X_3d
        triangulation_valid_mask[i] = True
        rejection_reasons.append("Valid")

    # 4. Reprojection checks using original camera matrices
    P_a_orig = K_a @ T_cw_a[:3, :]
    P_b_orig = K_b @ T_cw_b[:3, :]
    P_c_orig = K_c @ T_cw_c[:3, :]

    def project_pts(P, pts):
        homog = (P[:, :3] @ pts.T + P[:, 3:4]).T
        return homog[:, :2] / homog[:, 2:3]

    res_a = np.full(n_pts, np.nan)
    res_b = np.full(n_pts, np.nan)
    res_c = np.full(n_pts, np.nan)
    final_valid_mask = np.zeros(n_pts, dtype=bool)

    valid_idx = np.where(triangulation_valid_mask)[0]
    if len(valid_idx) > 0:
        proj_a = project_pts(P_a_orig, triangulated_pts[valid_idx])
        proj_b = project_pts(P_b_orig, triangulated_pts[valid_idx])
        proj_c = project_pts(P_c_orig, triangulated_pts[valid_idx])

        r_a = np.linalg.norm(proj_a - pts_a[valid_idx], axis=-1)
        r_b = np.linalg.norm(proj_b - pts_b[valid_idx], axis=-1)
        r_c = np.linalg.norm(proj_c - pts_c[valid_idx], axis=-1)

        res_a[valid_idx] = r_a
        res_b[valid_idx] = r_b
        res_c[valid_idx] = r_c

        per_point_max_res = np.maximum(np.maximum(r_a, r_b), r_c)
        passed_local = per_point_max_res < max_reproj_error
        final_valid_mask[valid_idx] = passed_local
        for local_index, original_index in enumerate(valid_idx):
            if not passed_local[local_index]:
                rejection_reasons[original_index] = (
                    f"Reprojection error fails strict <{max_reproj_error}px limit "
                    f"(max error: {per_point_max_res[local_index]:.6g}px)"
                )

    # 5. Direct-head comparison with cheirality masking (excluding invalid from stats)
    dir_cam_a = (R_a @ pts_3d_direct.T + t_a[:, None]).T
    dir_cam_b = (R_b @ pts_3d_direct.T + t_b[:, None]).T
    dir_cam_c = (R_c @ pts_3d_direct.T + t_c[:, None]).T
    dir_cheirality_mask = (dir_cam_a[:, 2] > 0) & (dir_cam_b[:, 2] > 0) & (dir_cam_c[:, 2] > 0)

    stat_mask = triangulation_valid_mask & dir_cheirality_mask
    if np.any(stat_mask):
        mean_head_3d_div = float(np.mean(np.linalg.norm(pts_3d_direct[stat_mask] - triangulated_pts[stat_mask], axis=-1)))
    else:
        mean_head_3d_div = float('nan')

    dir_res_a = np.full(n_pts, np.nan)
    dir_res_b = np.full(n_pts, np.nan)
    dir_res_c = np.full(n_pts, np.nan)

    if np.any(dir_cheirality_mask):
        valid_dir_pts = pts_3d_direct[dir_cheirality_mask]
        d_idx = np.where(dir_cheirality_mask)[0]
        
        proj_da = project_pts(P_a_orig, valid_dir_pts)
        proj_db = project_pts(P_b_orig, valid_dir_pts)
        proj_dc = project_pts(P_c_orig, valid_dir_pts)

        dir_res_a[d_idx] = np.linalg.norm(proj_da - pts_a[d_idx], axis=-1)
        dir_res_b[d_idx] = np.linalg.norm(proj_db - pts_b[d_idx], axis=-1)
        dir_res_c[d_idx] = np.linalg.norm(proj_dc - pts_c[d_idx], axis=-1)

    max_reproj_val = float(np.nanmax(np.maximum(np.maximum(res_a, res_b), res_c))) if np.any(final_valid_mask) or np.any(triangulation_valid_mask) else float('inf')

    return {
        "triangulation_valid_mask": triangulation_valid_mask,
        "final_valid_mask": final_valid_mask,
        "triangulated_points": triangulated_pts,
        "per_point_rejection_reasons": rejection_reasons,
        "reprojection_errors_a": res_a,
        "reprojection_errors_b": res_b,
        "reprojection_errors_c": res_c,
        "max_reprojection_error": max_reproj_val,
        "all_per_point_under_2px": bool(np.all(final_valid_mask)) and bool(np.all(np.nan_to_num(np.maximum(np.maximum(res_a, res_b), res_c), nan=float('inf')) < max_reproj_error)),
        "mean_head_3d_divergence": mean_head_3d_div,
        "direct_head_reprojection_errors": {
            "view_a": dir_res_a,
            "view_b": dir_res_b,
            "view_c": dir_res_c
        },
        "uncertainty_status": "unavailable_from_vggt_raw_heads (explicitly uncalibrated)",
        "epistemic_note": "Passing withheld-view tests supports consistency for tested tracks but does not prove universal root-cause or change detection."
    }
