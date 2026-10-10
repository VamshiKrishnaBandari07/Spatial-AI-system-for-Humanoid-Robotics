"""Independent RGB-D coordinate checks; these do not measure model accuracy."""

import numpy as np
from .public_data import valid_transform


def unproject(depth, K, stride=1):
    depth = np.asarray(depth, dtype=float)
    K = np.asarray(K, dtype=float)
    if depth.ndim != 2 or K.shape != (3, 3) or not np.isfinite(K).all():
        raise ValueError("Expected depth image and finite 3x3 intrinsics")
    if stride < 1 or K[0, 0] <= 0 or K[1, 1] <= 0:
        raise ValueError("Positive stride and focal lengths required")
    if not np.allclose(K[2], [0, 0, 1]) or not np.allclose(K[[0, 1], [1, 0]], 0):
        raise ValueError("Only zero-skew pinhole intrinsics are supported")
    y, x = np.mgrid[0 : depth.shape[0] : stride, 0 : depth.shape[1] : stride]
    z = depth[y, x]
    mask = np.isfinite(z) & (z > 0)
    pixels = np.column_stack([x[mask], y[mask]])
    rays = np.column_stack([pixels, np.ones(len(pixels))]) @ np.linalg.inv(K).T
    return rays * z[mask, None], pixels


def project(points, K):
    points = np.asarray(points, dtype=float)
    K = np.asarray(K, dtype=float)
    if points.ndim != 2 or points.shape[1] != 3 or K.shape != (3, 3):
        raise ValueError("Expected Nx3 camera points and 3x3 intrinsics")
    if (
        not np.isfinite(points).all()
        or not np.isfinite(K).all()
        or np.any(points[:, 2] <= 0)
    ):
        raise ValueError("Projection requires finite points in front of camera")
    homogeneous = points @ K.T
    return homogeneous[:, :2] / homogeneous[:, 2, None]


def transform_points(points, T_destination_source):
    T = valid_transform(T_destination_source)
    points = np.asarray(points, dtype=float)
    if points.ndim != 2 or points.shape[1] != 3 or not np.isfinite(points).all():
        raise ValueError("Expected finite Nx3 points")
    return points @ T[:3, :3].T + T[:3, 3]


def depth_consistency(source, target, stride=8):
    """Nearest-pixel cross-view sensor-depth residuals, not calibrated uncertainty.

    Reject occluded projected samples and never assume unregistered RGB is depth.
    """
    if not source["rgb_depth_aligned"] or not target["rgb_depth_aligned"]:
        raise ValueError("Cross-view RGB-D checks require verified aligned calibration")
    points, _ = unproject(source["depth_metres"], source["K"], stride)
    world = transform_points(points, source["T_world_camera"])
    camera = transform_points(world, np.linalg.inv(target["T_world_camera"]))
    camera = camera[camera[:, 2] > 0]
    uv = np.rint(project(camera, target["K"])).astype(int)
    height, width = target["depth_metres"].shape
    inside = (
        (uv[:, 0] >= 0) & (uv[:, 0] < width) & (uv[:, 1] >= 0) & (uv[:, 1] < height)
    )
    uv, camera = uv[inside], camera[inside]
    observed = target["depth_metres"][uv[:, 1], uv[:, 0]]
    valid = np.isfinite(observed) & (observed > 0)
    occluded = valid & (camera[:, 2] > observed + 0.05)
    usable = valid & ~occluded
    residuals = np.abs(camera[usable, 2] - observed[usable])
    return {
        "projected_samples": len(camera),
        "valid_target_samples": int(valid.sum()),
        "occluded_samples_excluded": int(occluded.sum()),
        "usable_samples": len(residuals),
        "median_absolute_depth_residual_metres": float(np.median(residuals))
        if len(residuals)
        else None,
        "p90_absolute_depth_residual_metres": float(np.quantile(residuals, 0.9))
        if len(residuals)
        else None,
        "interpretation": "Native sensor/pose consistency diagnostic, nearest-pixel samples; not model reconstruction accuracy or calibrated uncertainty",
    }
