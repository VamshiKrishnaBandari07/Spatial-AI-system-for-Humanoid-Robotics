"""Robust point-only bundle adjustment with fixed, independently checked cameras."""

from dataclasses import dataclass
import numpy as np
from numpy.typing import NDArray
from scipy.optimize import least_squares


@dataclass(frozen=True)
class ReprojectionFit:
    point: NDArray[np.float64]
    errors_px: NDArray[np.float64]
    accepted: bool
    covariance: NDArray[np.float64] | None
    initial_errors_px: NDArray[np.float64]


def refine_point(
    initial: NDArray[np.float64], pixels: NDArray[np.float64],
    intrinsics: NDArray[np.float64], poses: NDArray[np.float64],
    sigma_px: NDArray[np.float64] | None = None, limit_px: float = 2.0,
) -> ReprojectionFit:
    """Fit one world point (3,) to Vx2 pixels, Vx3x3 K and Vx4x4 T_camera_world.

    Huber residuals are normalized by measurement uncertainty. Uncertainty
    changes optimization weights, never the final unweighted pixel limit.
    Every supplied view must satisfy positive depth and the strict limit.
    Cameras remain fixed: optimizing them on an object could conceal bad
    correspondences or disturb static-scene registration. Surface keypoints
    are optimized individually; their centroid is not a corresponding pixel.
    """
    initial = np.asarray(initial, dtype=float)
    pixels = np.asarray(pixels, dtype=float)
    intrinsics = np.asarray(intrinsics, dtype=float)
    poses = np.asarray(poses, dtype=float)
    n = len(pixels)
    sigma = np.ones(n) if sigma_px is None else np.asarray(sigma_px, dtype=float)
    if (initial.shape != (3,) or pixels.shape != (n, 2) or n < 2
            or intrinsics.shape != (n, 3, 3) or poses.shape != (n, 4, 4)
            or sigma.shape != (n,) or not np.isfinite(limit_px) or limit_px <= 0
            or not all(np.isfinite(x).all() for x in (initial, pixels, intrinsics, poses, sigma))
            or np.any(sigma <= 0)):
        raise ValueError("Invalid multi-view point refinement inputs")

    if (np.any(intrinsics[:, 0, 0] <= 0) or np.any(intrinsics[:, 1, 1] <= 0)
            or not np.allclose(intrinsics[:, 2], [0, 0, 1])
            or not np.allclose(poses[:, 3], [0, 0, 0, 1])
            or not np.allclose(poses[:, :3, :3] @ np.swapaxes(poses[:, :3, :3], 1, 2), np.eye(3), atol=1e-3)
            or np.any(np.linalg.det(poses[:, :3, :3]) <= 0)):
        raise ValueError("Invalid pinhole intrinsics or right-handed rigid pose")

    def projection(point: NDArray[np.float64]) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        camera = np.einsum("vij,j->vi", poses[:, :3, :3], point) + poses[:, :3, 3]
        image = np.einsum("vij,vj->vi", intrinsics, camera)
        z = image[:, 2]
        safe_z = np.where(np.abs(z) < 1e-12, 1e-12, z)
        return image[:, :2] / safe_z[:, None], camera[:, 2]

    def residual(point: NDArray[np.float64]) -> NDArray[np.float64]:
        uv, _ = projection(point)
        return ((uv - pixels) / sigma[:, None]).ravel()

    initial_uv, _ = projection(initial)
    fit = least_squares(residual, initial, loss="huber", f_scale=1.0,
                        method="trf", max_nfev=60)
    # Huber can favour one camera when a residual exceeds its robust scale.
    # Once a point is fitted, compare a symmetric least-squares polish using
    # the same measurements. Choose only a positive-depth solution with a
    # smaller worst *unweighted* pixel error. Final acceptance is unchanged.
    balanced = least_squares(residual, fit.x, loss="linear", method="trf", max_nfev=30)
    robust_uv, _ = projection(fit.x)
    balanced_uv, balanced_depth = projection(balanced.x)
    if (balanced.success and np.all(balanced_depth > 0)
            and np.linalg.norm(balanced_uv-pixels, axis=1).max()
            < np.linalg.norm(robust_uv-pixels, axis=1).max()):
        fit = balanced
    uv, depths = projection(fit.x)
    errors = np.linalg.norm(uv - pixels, axis=1)
    accepted = bool(fit.success and np.isfinite(fit.x).all()
                    and np.all(depths > 0) and np.all(errors < limit_px))
    information = fit.jac.T @ fit.jac
    accepted = accepted and np.linalg.matrix_rank(information) == 3
    covariance = None
    if np.linalg.matrix_rank(information) == 3:
        covariance = np.linalg.inv(information) * max(1.0, float(np.mean(fit.fun ** 2)))
    return ReprojectionFit(fit.x, errors, accepted, covariance,
                           np.linalg.norm(initial_uv - pixels, axis=1))
