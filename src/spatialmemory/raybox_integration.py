"""Real surface adapter. Missing uncertainty never becomes a guessed bound."""
from __future__ import annotations
import math
import numpy as np
from .object_visibility import evaluate_box_view


def conformal_depth_bound(frame_max_errors: np.ndarray, alpha: float) -> float:
    """Split-conformal absolute depth bound from HELD-OUT frame-level maxima.

    Scores must come from independently calibrated reference depth, in the target
    scale, on exchangeable held-out frames. Pixels from one frame are not treated
    as independent samples. Coverage is marginal for a new exchangeable frame,
    not distribution-free across new cameras/domains and not deterministic.
    The caller must establish provenance, independence and applicability.
    """
    scores = np.asarray(frame_max_errors, float)
    if scores.ndim != 1 or not len(scores) or not np.isfinite(scores).all() or np.any(scores < 0):
        raise ValueError('Expected finite nonnegative held-out frame maxima')
    if not 0 < alpha < 1:
        raise ValueError('alpha must be between zero and one')
    rank = math.ceil((len(scores) + 1) * (1 - alpha))
    if rank > len(scores):
        raise ValueError('Insufficient independent calibration frames for finite bound')
    return float(np.partition(scores, rank - 1)[rank - 1])


def propagated_geometry_bound(corners, transform, camera, *, localization_radius,
                              scale_error, registration_angle, registration_translation,
                              camera_angle, camera_translation) -> float:
    """Conservative Euclidean perturbation radius, also bounding camera-Z error.

    Absolute scale error; rotation errors in radians; distances in corresponding
    world units. Bounds must be supplied from validated calibration/estimation,
    not registration RMSE or neural confidence. Rotations satisfy ||R-Rhat|| <=
    2 sin(theta/2). This scalar does not model which pixels change under lateral
    uncertainty, so the resulting ray classification remains diagnostic only.
    """
    values = [localization_radius, scale_error, registration_angle,
              registration_translation, camera_angle, camera_translation]
    if not np.isfinite(values).all() or min(values) < 0 or max(registration_angle, camera_angle) > np.pi:
        raise ValueError('Invalid perturbation bounds')
    points = np.asarray(corners, float)
    if points.ndim != 2 or points.shape[1] != 3 or not np.isfinite(points).all():
        raise ValueError('Expected finite Nx3 corners')
    from .object_visibility import _transform
    t = _transform(transform, 'object_to_target', True)
    _transform(camera, 'camera', False)
    scale = np.cbrt(np.linalg.det(t[:3, :3]))
    radius = np.linalg.norm(points, axis=1).max()
    registration_error = ((scale + scale_error) * localization_radius + scale_error * radius
                          + 2 * scale * radius * np.sin(registration_angle / 2)
                          + registration_translation)
    target = points @ t[:3, :3].T + t[:3, 3]
    return float(registration_error + 2 * np.linalg.norm(target, axis=1).max()
                 * np.sin(camera_angle / 2) + camera_translation)


def raybox_view(extent, surface, pose, support, registration=None) -> dict:
    """Adapt shared BEFORE-world object box to a real target reconstruction view.

    Repository registration maps AFTER -> BEFORE. Its inverse is required when
    target surfaces/poses remain in AFTER. Depth and intrinsics are consumed at
    their native reconstruction grid; no RGB resize assumption is made.
    Optional surface['raybox_uncertainty'] must contain depth_error_bound,
    geometry_error_bound, scale_id and source. It is caller supplied, not inferred
    from the checkpoint confidence. A metric flag alone supplies no uncertainty.
    """
    if extent is None:
        return {'status': 'insufficient', 'reason': 'missing_object_extent'}
    transform = np.eye(4)
    if registration is not None:
        if not registration.valid:
            return {'status': 'insufficient', 'reason': 'invalid_registration'}
        after_to_before = np.eye(4)
        after_to_before[:3, :3] = registration.scale * np.asarray(registration.rotation)
        after_to_before[:3, 3] = registration.translation
        transform = np.linalg.inv(after_to_before)
    audit = {'object_to_target_world': transform.tolist(),
             'target_world_to_camera': np.asarray(pose).tolist(),
             'depth_convention': 'camera_z', 'depth_shape': list(surface['depth'].shape),
             'promotes_change_decision': False}
    bounds = surface.get('raybox_uncertainty')
    if not bounds or not bounds.get('source') or not bounds.get('scale_id'):
        return {**audit, 'status': 'insufficient', 'reason': 'missing_validated_uncertainty_bounds'}
    depth_bound = np.asarray(bounds['depth_error_bound'], float)
    if depth_bound.ndim == 0:
        depth_bound = np.full(surface['depth'].shape, float(depth_bound))
    result = evaluate_box_view(
        extent[0], extent[1], transform, pose, surface['intrinsics'],
        surface['depth'], np.asarray(support, bool), depth_bound,
        bounds['geometry_error_bound'], geometry_scale_id=bounds['scale_id'],
        depth_scale_id=bounds['scale_id'], pixel_stride=1)
    hit = result.intersects
    # Limit JSON size while retaining real diagnostic rays for UI inspection.
    indices = np.flatnonzero(hit)
    selected = indices[np.linspace(0, len(indices)-1, min(64, len(indices))).astype(int)] if len(indices) else []
    return {**audit, 'status': 'diagnostic_only', 'uncertainty_source': bounds['source'],
            'counts': result.counts, 'fractions': result.fractions,
            'projected_region_xyxy': result.projected_region,
            'touches_image_boundary': result.touches_image_boundary,
            'coverage': result.coverage,
            'rays': [{'pixel_uv': result.pixels_uv[i].tolist(),
                      'near_z': float(result.near_z[i]), 'far_z': float(result.far_z[i]),
                      'state': int(result.states[i])} for i in selected]}
