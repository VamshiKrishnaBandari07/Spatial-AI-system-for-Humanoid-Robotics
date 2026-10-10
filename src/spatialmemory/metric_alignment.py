"""Gated metric registration: explicit initial transform, trimmed coarse-to-fine ICP."""

import numpy as np
from scipy.spatial import cKDTree

from .public_data import valid_transform
from .rgbd_validation import transform_points


def downsample(points, voxel):
    points = np.asarray(points, float)
    if points.ndim != 2 or points.shape[1] != 3 or not np.isfinite(points).all():
        raise ValueError("Finite Nx3 clouds required")
    _, ids = np.unique(
        np.floor(points / voxel).astype(np.int64), axis=0, return_index=True
    )
    return points[ids]


def alignment_metrics(source, target, T, threshold):
    transformed = transform_points(source, T)
    forward = cKDTree(target).query(transformed)[0]
    backward = cKDTree(transformed).query(target)[0]
    good = forward < threshold
    reverse = backward < threshold
    return {
        "source_points": len(source),
        "target_points": len(target),
        "fitness": float(good.mean()),
        "reverse_fitness": float(reverse.mean()),
        "symmetric_overlap": float(min(good.mean(), reverse.mean())),
        "inliers": int(good.sum()),
        "inlier_rmse_metres": float(np.sqrt(np.mean(forward[good] ** 2)))
        if good.any()
        else None,
        "median_distance_metres": float(np.median(forward)),
        "threshold_metres": threshold,
    }


def metric_register(source, target, initial, threshold=0.08, min_overlap=0.35):
    """No identity fallback. Initial transform must be explicitly sourced by caller.

    Every fifth voxel sample is held out from refinement. This is a geometric
    gate, not an independent reconstruction benchmark or a calibrated covariance.
    """
    T = valid_transform(initial).copy()
    source = downsample(source, 0.02)
    target = downsample(target, 0.02)
    if min(len(source), len(target)) < 30:
        raise ValueError("Insufficient metric cloud support")
    if min(np.linalg.matrix_rank(p - p.mean(0)) for p in [source, target]) < 2:
        raise ValueError("Degenerate metric cloud")
    source_test, target_test = source[::5], target[::5]
    source_train = source[np.arange(len(source)) % 5 != 0]
    target_train = target[np.arange(len(target)) % 5 != 0]
    history = []
    for voxel, radius in [(0.08, 0.24), (0.04, 0.12), (0.02, threshold)]:
        a, b = downsample(source_train, voxel), downsample(target_train, voxel)
        tree = cKDTree(b)
        for _ in range(25):
            moved = transform_points(a, T)
            distance, ids = tree.query(moved)
            eligible = distance < radius
            if eligible.sum() < 20:
                break
            cutoff = np.quantile(distance[eligible], 0.8)
            keep = eligible & (distance <= cutoff)
            x, y = moved[keep], b[ids[keep]]
            xc, yc = x.mean(0), y.mean(0)
            if np.linalg.matrix_rank(x - xc) < 2:
                break
            U, _, Vt = np.linalg.svd((x - xc).T @ (y - yc))
            D = np.eye(3)
            D[-1, -1] = np.linalg.det(Vt.T @ U.T)
            R = Vt.T @ D @ U.T
            delta = np.eye(4)
            delta[:3, :3] = R
            delta[:3, 3] = yc - R @ xc
            T = delta @ T
            if np.linalg.norm(delta - np.eye(4)) < 1e-6:
                break
        history.append(alignment_metrics(a, b, T, radius))
    metrics = alignment_metrics(source, target, T, threshold)
    heldout = alignment_metrics(source_test, target_test, T, threshold)
    initial_metrics = alignment_metrics(source, target, initial, threshold)
    correction = T @ np.linalg.inv(initial)
    correction_angle = float(
        np.degrees(np.arccos(np.clip((np.trace(correction[:3, :3]) - 1) / 2, -1, 1)))
    )
    correction_translation = float(np.linalg.norm(correction[:3, 3]))
    valid = (
        metrics["symmetric_overlap"] >= min_overlap
        and heldout["symmetric_overlap"] >= min_overlap
        and metrics["inlier_rmse_metres"] is not None
        and metrics["inlier_rmse_metres"] <= threshold * 0.6
        and heldout["inlier_rmse_metres"] is not None
        and heldout["inlier_rmse_metres"] <= threshold * 0.6
        and correction_translation <= 0.15
        and correction_angle <= 10
    )
    return {
        "valid": bool(valid),
        "T_reference_rescan": T.tolist(),
        "initial": initial_metrics,
        "refined": metrics,
        "heldout": heldout,
        "history": history,
        "correction_translation_metres": correction_translation,
        "correction_rotation_degrees": correction_angle,
        "reason": "Metric overlap, RMSE, held-out geometry and correction gates passed"
        if valid
        else "Metric registration gate failed",
        "estimation_condition": "Explicit source initial transform plus robust ICP; not autonomous global registration",
    }
