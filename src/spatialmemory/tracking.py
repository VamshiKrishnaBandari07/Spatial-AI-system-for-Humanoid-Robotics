"""Sparse optical-flow tracks with forward/backward checks and homography compensation."""

import cv2, numpy as np


def track(previous, current, points, max_error=1.0):
    points = np.asarray(points, dtype=np.float32).reshape(-1, 1, 2)
    if not len(points):
        return {
            "points": [],
            "valid": [],
            "residual_motion": [],
            "motion_is_physical": False,
        }
    pg = cv2.cvtColor(previous, cv2.COLOR_BGR2GRAY)
    cg = cv2.cvtColor(current, cv2.COLOR_BGR2GRAY)
    # Keep the coarsest image large enough for the LK support window.
    levels = max(0, min(3, int(np.floor(np.log2(min(pg.shape) / 42)))))
    nxt, status, _ = cv2.calcOpticalFlowPyrLK(pg, cg, points, None, maxLevel=levels)
    if nxt is None:
        raise ValueError("Optical flow failed")
    back, back_status, _ = cv2.calcOpticalFlowPyrLK(cg, pg, nxt, None, maxLevel=levels)
    if back is None:
        raise ValueError("Backward optical flow failed")
    valid = (
        (status[:, 0] > 0)
        & (back_status[:, 0] > 0)
        & (np.linalg.norm(back[:, 0] - points[:, 0], axis=1) < max_error)
    )
    residual = np.full((len(points), 2), np.nan)
    if valid.sum() >= 8:
        H, inliers = cv2.findHomography(points[valid], nxt[valid], cv2.RANSAC, 2.0)
        if H is not None:
            predicted = cv2.perspectiveTransform(points, H)[:, 0]
            residual = nxt[:, 0] - predicted
    return {
        "points": nxt[:, 0].tolist(),
        "valid": valid.tolist(),
        "residual_motion": residual.tolist(),
        "motion_is_physical": False,
        "warning": "Homography residual also includes parallax and model error",
    }
