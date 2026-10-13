"""Object-specific expected-region coverage, independent of observed occupancy."""
from __future__ import annotations
import numpy as np
from .object_visibility import _transform


def object_view_coverage(points, intrinsics, world_to_camera, image_shape,
                         valid_pixels=None) -> dict:
    """Project Nx3 target-world support points onto the matching camera grid.

    Ranking is geometric coverage, not visibility or confidence. Behind-camera
    and padding samples remain in the denominator. Missing/changed depth never
    suppresses a candidate view: that is tested later by occupancy reasoning.
    Image-region bounds are xyxy, exclusive. Incomplete object samples do not
    certify the full object silhouette or account for uncertain poses.
    """
    points = np.asarray(points, float)
    k = np.asarray(intrinsics, float)
    pose = _transform(world_to_camera, 'world_to_camera', False)
    if points.ndim != 2 or points.shape[1] != 3 or not len(points) or not np.isfinite(points).all():
        raise ValueError('Expected nonempty finite Nx3 object support')
    if k.shape != (3,3) or not np.isfinite(k).all() or k[0,0] <= 0 or k[1,1] <= 0 or not np.allclose(k[2],[0,0,1]):
        raise ValueError('Invalid intrinsics')
    h,w = image_shape
    if h <= 0 or w <= 0:
        raise ValueError('Invalid image shape')
    content = np.ones((h,w),bool) if valid_pixels is None else np.asarray(valid_pixels)
    if content.shape != (h,w) or content.dtype != bool:
        raise ValueError('Expected boolean content mask matching camera grid')
    cam = points @ pose[:3,:3].T + pose[:3,3]
    front = cam[:,2] > 0
    uv = np.full((len(points),2),np.nan)
    projected = cam[front] @ k.T
    uv[front] = projected[:,:2] / projected[:,2,None]
    inside = front & np.isfinite(uv).all(axis=1)
    inside &= (uv[:,0] >= 0) & (uv[:,0] < w) & (uv[:,1] >= 0) & (uv[:,1] < h)
    ids = np.flatnonzero(inside)
    pixels = np.floor(uv[ids]).astype(int)
    inside[ids] &= content[pixels[:,1],pixels[:,0]]
    hits = uv[inside]
    region = None
    if len(hits):
        lo = np.floor(hits.min(axis=0)).astype(int)
        hi = np.minimum(np.floor(hits.max(axis=0)).astype(int)+1,[w,h])
        region = [int(lo[0]),int(lo[1]),int(hi[0]),int(hi[1])]
    return {'in_frame_fraction':float(inside.mean()),'support_points':len(points),
            'in_frame_points':int(inside.sum()),'region_xyxy':region,
            'projected_area_pixels': (region[2]-region[0])*(region[3]-region[1]) if region else 0,
            'occupancy_used_for_selection':False,'physical_visibility_verified':False,
            'coordinate_grid':'target reconstruction depth/intrinsics grid'}


def rank_object_views(views: list[dict]) -> list[str]:
    """Rank usable projections by coverage then projected resolution, deterministically."""
    usable = [v for v in views if v['object_projection']['in_frame_points'] > 0]
    usable.sort(key=lambda v: (-v['object_projection']['in_frame_fraction'],
                              -v['object_projection']['projected_area_pixels'],v['frame_id']))
    return [v['frame_id'] for v in usable]
