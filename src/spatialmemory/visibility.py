"""Visibility requires measured depth, not holes in a sparse point cloud."""

import numpy as np
from .geometry import project


def visibility(points, depth, K, pose, tolerance=0.05):
    uv, z = project(np.asarray(points), K, pose)
    result = []
    for (u, v), distance in zip(uv, z):
        if not np.isfinite([u, v, distance]).all() or distance <= 0:
            result.append("UNKNOWN")
            continue
        x, y = int(round(u)), int(round(v))
        if not 0 <= y < depth.shape[0] or not 0 <= x < depth.shape[1]:
            result.append("UNKNOWN")
            continue
        measured = float(depth[y, x])
        if not np.isfinite(measured) or measured <= 0:
            result.append("UNKNOWN")
        elif measured < distance - tolerance:
            result.append("OCCLUDED")
        elif measured > distance + tolerance:
            result.append("FREE")
        else:
            result.append("OCCUPIED")
    return result
