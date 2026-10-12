"""Retain actual computed object surfaces, not fabricated box-corner geometry."""

import numpy as np
from numpy.typing import NDArray


def sample_surface(points: NDArray[np.float64], limit: int = 32) -> list[list[float]]:
    """Bounded deterministic samples of Nx3 world points; no interpolation."""
    points = np.asarray(points,float)
    if points.shape == (0,):
        return []
    if points.ndim != 2 or points.shape[1] != 3 or not np.isfinite(points).all() or limit < 1:
        raise ValueError("Finite Nx3 observed surface points required")
    if not len(points):
        return []
    indices = np.unique(np.linspace(0,len(points)-1,min(limit,len(points))).round().astype(int))
    return points[indices].tolist()
