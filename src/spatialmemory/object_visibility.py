"""Camera-Z ray/box evidence. NumPy only; no identity or removal inference.

Transforms: BEFORE --T_after_before (Sim3)--> AFTER --T_camera_after (SE3)--> camera.
Camera convention: X right, Y down, Z forward. Pixel centers have integer (u,v)
coordinates. Depth must be undistorted camera-Z, aligned to K's image grid.
The caller certifies a shared AFTER-world/depth scale via matching nonempty IDs.
Error bounds are supplied conservative absolute camera-Z bounds, not probabilities.
"""
from dataclasses import dataclass
from enum import IntEnum
import numpy as np


class EvidenceState(IntEnum):
    INSUFFICIENT = 0
    FREE = 1
    OCCUPIED = 2
    OCCLUDED = 3


@dataclass(frozen=True)
class ViewEvidence:
    pixels_uv: np.ndarray          # (N,2) integer sampled pixel centers
    intersects: np.ndarray         # (N,) rays intersect forward clipped volume
    near_z: np.ndarray             # (N,) camera-Z; NaN for no intersection
    far_z: np.ndarray              # (N,) camera-Z; NaN for no intersection
    states: np.ndarray             # (N,) EvidenceState integer codes
    observed_intervals: np.ndarray # (N,2), NaN for invalid depth
    possible_intervals: np.ndarray # (N,2) [near-error,far+error]
    certain_intervals: np.ndarray  # (N,2) [near+error,far-error], can be empty
    projected_region: tuple[int, int, int, int] | None  # sampled bounds xyxy exclusive
    touches_image_boundary: bool  # warning, not a measure of full-volume coverage
    counts: dict[str, int]         # intersection rays only
    fractions: dict[str, float]    # denominator: all intersection rays incl invalid
    coverage: str                 # explicitly not a full visibility guarantee


def _transform(value: np.ndarray, name: str, similarity: bool) -> np.ndarray:
    t = np.asarray(value, dtype=float)
    if t.shape != (4, 4) or not np.isfinite(t).all():
        raise ValueError(f'{name}: expected finite 4x4 matrix')
    if not np.allclose(t[3], [0, 0, 0, 1], atol=1e-8, rtol=0):
        raise ValueError(f'{name}: invalid homogeneous row')
    a = t[:3, :3]
    scale = np.cbrt(np.linalg.det(a))
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError(f'{name}: requires positive scale and proper rotation')
    if not similarity and not np.isclose(scale, 1, atol=1e-7, rtol=0):
        raise ValueError(f'{name}: camera transform must be rigid')
    rotation = a / scale
    if not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-7, rtol=0):
        raise ValueError(f'{name}: shear/nonuniform scale unsupported')
    return t


def evaluate_box_view(
    before_box_min: np.ndarray, before_box_max: np.ndarray,
    T_after_before: np.ndarray, T_camera_after: np.ndarray, K: np.ndarray,
    depth: np.ndarray, depth_valid: np.ndarray, depth_error_bound: np.ndarray,
    geometry_error_bound: float, *, geometry_scale_id: str, depth_scale_id: str,
    depth_convention: str = 'camera_z', pixel_stride: int = 1,
    near_clip: float = 1e-6,
) -> ViewEvidence:
    """Intersect native camera rays with original BEFORE AABB using slab intervals.

    Each ray is K^-1[u,v,1] normalized to z=1, hence parameter t is camera Z.
    Inverse composed transforms map origin and direction to BEFORE without
    renormalizing direction: Sim3 scale is retained and t stays camera-Z.

    FREE: observed interval lies strictly beyond far+error.
    OCCLUDED: observed interval lies strictly in front of near-error.
    OCCUPIED: observed interval lies strictly inside [near+error,far-error].
    Otherwise INSUFFICIENT. Equality is conservatively insufficient.
    Pixels outside the projected volume are INSUFFICIENT, excluded from counts.

    Bounds do not correct lateral pose errors: caller must provide a conservative
    spatial volume and valid rays. This model alone does not certify physical
    visibility or establish that a complete object's footprint is in the image.
    """
    lo, hi = np.asarray(before_box_min, float), np.asarray(before_box_max, float)
    if lo.shape != (3,) or hi.shape != (3,) or not np.isfinite([lo, hi]).all() or np.any(lo >= hi):
        raise ValueError('Expected finite ordered three-dimensional bounds')
    registration = _transform(T_after_before, 'T_after_before', True)
    camera = _transform(T_camera_after, 'T_camera_after', False)
    k = np.asarray(K, float)
    if (k.shape != (3, 3) or not np.isfinite(k).all() or k[0, 0] <= 0 or k[1, 1] <= 0
            or not np.allclose(k[2], [0, 0, 1], atol=1e-8, rtol=0)
            or abs(np.linalg.det(k)) < 1e-12):
        raise ValueError('Invalid pinhole intrinsics')
    if (not isinstance(geometry_scale_id, str) or not geometry_scale_id.strip()
            or geometry_scale_id != depth_scale_id):
        raise ValueError('Explicit matching geometry/depth scale IDs required')
    if depth_convention != 'camera_z':
        raise ValueError('Only aligned camera_z depth supported; convert ray distance first')
    d, valid, err = np.asarray(depth, float), np.asarray(depth_valid), np.asarray(depth_error_bound, float)
    if d.ndim != 2 or not d.size or valid.shape != d.shape or err.shape != d.shape or valid.dtype != bool:
        raise ValueError('Depth, bool validity and error arrays must share nonempty (H,W) shape')
    if not np.isfinite(err).all() or np.any(err < 0):
        raise ValueError('Depth error bounds must be finite and nonnegative')
    if not np.isfinite(geometry_error_bound) or geometry_error_bound < 0:
        raise ValueError('Geometry error bound must be finite and nonnegative')
    if not isinstance(pixel_stride, int) or isinstance(pixel_stride, bool) or pixel_stride < 1:
        raise ValueError('pixel_stride must be a positive integer')
    if not np.isfinite(near_clip) or near_clip <= 0:
        raise ValueError('near_clip must be positive in depth units')
    h, w = d.shape
    # Include last row/column for boundary diagnostics even with subsampling.
    us = np.unique(np.r_[np.arange(0, w, pixel_stride), w - 1])
    vs = np.unique(np.r_[np.arange(0, h, pixel_stride), h - 1])
    u, v = np.meshgrid(us, vs)
    pixels = np.column_stack([u.ravel(), v.ravel()])
    rays = np.column_stack([pixels, np.ones(len(pixels))]) @ np.linalg.inv(k).T
    rays /= rays[:, 2, None]
    inverse = np.linalg.inv(camera @ registration)
    origin = inverse[:3, 3]
    directions = rays @ inverse[:3, :3].T
    entry, exit = np.full(len(rays), -np.inf), np.full(len(rays), np.inf)
    possible = np.ones(len(rays), bool)
    for axis in range(3):
        direction = directions[:, axis]
        parallel = direction == 0  # Do not discard small nonzero components.
        possible &= ~(parallel & ((origin[axis] < lo[axis]) | (origin[axis] > hi[axis])))
        a = np.full(len(rays), -np.inf)
        b = np.full(len(rays), np.inf)
        nonparallel = ~parallel
        first = (lo[axis] - origin[axis]) / direction[nonparallel]
        second = (hi[axis] - origin[axis]) / direction[nonparallel]
        a[nonparallel], b[nonparallel] = np.minimum(first, second), np.maximum(first, second)
        entry, exit = np.maximum(entry, a), np.minimum(exit, b)
    entry = np.maximum(entry, near_clip)
    hit = possible & np.isfinite(entry) & np.isfinite(exit) & (exit > entry)
    near, far = np.where(hit, entry, np.nan), np.where(hit, exit, np.nan)
    samples, errors = d[v.ravel(), u.ravel()], err[v.ravel(), u.ravel()]
    usable = valid[v.ravel(), u.ravel()] & np.isfinite(samples) & (samples > 0) & (samples - errors > 0)
    observed = np.column_stack([samples - errors, samples + errors])
    observed[~usable] = np.nan
    outer = np.column_stack([near - geometry_error_bound, far + geometry_error_bound])
    inner = np.column_stack([near + geometry_error_bound, far - geometry_error_bound])
    states = np.full(len(rays), EvidenceState.INSUFFICIENT, np.uint8)
    ok = hit & usable
    states[ok & (observed[:, 0] > outer[:, 1])] = EvidenceState.FREE
    states[ok & (observed[:, 1] < outer[:, 0])] = EvidenceState.OCCLUDED
    states[ok & (observed[:, 0] > inner[:, 0]) & (observed[:, 1] < inner[:, 1])] = EvidenceState.OCCUPIED
    region, boundary = None, False
    if np.any(hit):
        xy = pixels[hit]
        region = (int(xy[:, 0].min()), int(xy[:, 1].min()), int(xy[:, 0].max()) + 1, int(xy[:, 1].max()) + 1)
        boundary = bool(np.any((xy[:, 0] == 0) | (xy[:, 0] == w - 1) | (xy[:, 1] == 0) | (xy[:, 1] == h - 1)))
    counts = {state.name: int(np.sum(states[hit] == state)) for state in EvidenceState}
    total = int(hit.sum())
    fractions = {name: count / total if total else 0.0 for name, count in counts.items()}
    return ViewEvidence(pixels, hit, near, far, states, observed, outer, inner,
                        region, boundary, counts, fractions,
                        'sampled_in_image_only; full object coverage not established')
