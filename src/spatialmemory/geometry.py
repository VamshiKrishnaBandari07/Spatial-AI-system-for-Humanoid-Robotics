"""Incremental sparse monocular SfM: essential initialization, PnP, triangulation."""

from dataclasses import dataclass, field
import cv2
import numpy as np
from scipy.optimize import least_squares


@dataclass
class Reconstruction:
    points: np.ndarray
    colors: np.ndarray
    poses: dict
    tracks: dict
    features: dict
    intrinsics: np.ndarray
    diagnostics: dict
    surfaces: dict = field(default_factory=dict)


def project(points, K, pose):
    p = (pose[:3, :3] @ points.T + pose[:3, 3, None]).T
    uv = (K @ p.T).T
    return uv[:, :2] / uv[:, 2, None], p[:, 2]


def extract(frames, camera=None):
    sift = cv2.SIFT_create(nfeatures=3000)
    result = {}
    for f in frames:
        im = cv2.imread(f.path)
        if camera is not None and camera.distortion_model == "opencv":
            im = cv2.undistort(
                im,
                camera.matrix(im.shape[1], im.shape[0]),
                np.array(camera.distortion_coefficients),
            )
        kp, desc = sift.detectAndCompute(cv2.cvtColor(im, cv2.COLOR_BGR2GRAY), None)
        result[f.id] = (np.array([k.pt for k in kp], dtype=np.float64), desc, im)
    return result


def match(a, b):
    if a is None or b is None or len(a) < 2 or len(b) < 2:
        return []
    matcher = cv2.BFMatcher(cv2.NORM_L2)
    forward = matcher.knnMatch(a, b, k=2)
    reverse = matcher.knnMatch(b, a, k=2)
    rev = {
        m.queryIdx: m.trainIdx
        for pair in reverse
        if len(pair) == 2
        for m, n in [pair]
        if m.distance < 0.75 * n.distance
    }
    return [
        (m.queryIdx, m.trainIdx)
        for pair in forward
        if len(pair) == 2
        for m, n in [pair]
        if m.distance < 0.75 * n.distance and rev.get(m.trainIdx) == m.queryIdx
    ]


def reconstruct(frames, config, camera=None):
    features = extract(frames, camera)
    im = features[frames[0].id][2]
    h, w = im.shape[:2]
    focal = config.focal_px or max(h, w)
    K = np.array([[focal, 0, w / 2], [0, focal, h / 2], [0, 0, 1.0]])
    if camera is not None:
        K = camera.matrix(w, h)
    points = []
    colors = []
    tracks = {}
    poses = {}
    residuals = []
    rejected = []
    recovered_anchors = []
    initialized = False
    for fa, fb in zip(frames, frames[1:]):
        if initialized and fa.id not in poses:
            # A failed/occluded frame must not invalidate all subsequent frames.
            # Reacquisition still requires real matches and robust PnP below.
            registered = [f for f in frames if f.index < fb.index and f.id in poses]
            if registered:
                fa = registered[-1]
                recovered_anchors.append({"frame_id": fb.id, "anchor_frame_id": fa.id})
        a, da, _ = features[fa.id]
        b, db, image = features[fb.id]
        pairs = match(da, db)
        if len(pairs) < 20:
            rejected.append([fb.id, "too few matches"])
            continue
        ia, ib = np.array(pairs).T
        pa = a[ia]
        pb = b[ib]
        if not initialized:
            E, mask = cv2.findEssentialMat(
                pa,
                pb,
                K,
                method=cv2.RANSAC,
                prob=0.999,
                threshold=config.reprojection_px,
            )
            if E is None or E.shape != (3, 3):
                continue
            count, R, t, mask = cv2.recoverPose(E, pa, pb, K, mask=mask)
            if count < 15:
                continue
            A = np.eye(4)
            B = np.eye(4)
            B[:3, :3] = R
            B[:3, 3] = t[:, 0]
            poses[fa.id] = A
            poses[fb.id] = B
            initialized = True
            valid = mask.ravel() > 0
        else:
            if fa.id not in poses:
                rejected.append([fb.id, "previous pose unavailable"])
                continue
            known = [
                (j, tracks[(fa.id, int(i))])
                for i, j in pairs
                if (fa.id, int(i)) in tracks
            ]
            if len(known) < 8:
                rejected.append([fb.id, "insufficient 3D tracks"])
                continue
            js, pids = np.array(known).T
            ok, rv, tv, inliers = cv2.solvePnPRansac(
                np.asarray(points)[pids],
                b[js],
                K,
                None,
                iterationsCount=200,
                reprojectionError=config.reprojection_px,
                confidence=0.999,
            )
            if not ok or inliers is None or len(inliers) < 8:
                rejected.append([fb.id, "PnP failed"])
                continue
            B = np.eye(4)
            B[:3, :3] = cv2.Rodrigues(rv)[0]
            B[:3, 3] = tv[:, 0]
            A = poses[fa.id]
            poses[fb.id] = B
            for j, pid in np.array(known)[inliers[:, 0]]:
                tracks[(fb.id, int(j))] = int(pid)
            valid = np.ones(len(pairs), bool)
        homogeneous = cv2.triangulatePoints(K @ A[:3], K @ B[:3], pa.T, pb.T).T
        with np.errstate(divide="ignore", invalid="ignore"):
            xyz = homogeneous[:, :3] / homogeneous[:, 3, None]
        ua, za = project(xyz, K, A)
        ub, zb = project(xyz, K, B)
        err = np.maximum(
            np.linalg.norm(ua - pa, axis=1), np.linalg.norm(ub - pb, axis=1)
        )
        ca = -A[:3, :3].T @ A[:3, 3]
        cb = -B[:3, :3].T @ B[:3, 3]
        va = xyz - ca
        vb = xyz - cb
        cosine = np.sum(va * vb, axis=1) / (
            np.linalg.norm(va, axis=1) * np.linalg.norm(vb, axis=1) + 1e-12
        )
        good = (
            valid
            & np.isfinite(xyz).all(axis=1)
            & (za > 0)
            & (zb > 0)
            & (err < config.reprojection_px)
            & (cosine < np.cos(np.deg2rad(1)))
        )
        for k in np.flatnonzero(good):
            ka = (fa.id, int(ia[k]))
            kb = (fb.id, int(ib[k]))
            if ka in tracks:
                tracks[kb] = tracks[ka]
                continue
            if kb in tracks:
                continue
            pid = len(points)
            points.append(xyz[k])
            x, y = np.rint(pb[k]).astype(int)
            colors.append(image[np.clip(y, 0, h - 1), np.clip(x, 0, w - 1), ::-1] / 255)
            tracks[ka] = pid
            tracks[kb] = pid
            residuals.append(float(err[k]))
    if len(points) < 20 or len(poses) < 2:
        raise ValueError(
            "Sparse reconstruction failed: insufficient parallax, texture, or consistent poses"
        )
    return Reconstruction(
        np.array(points),
        np.array(colors),
        poses,
        tracks,
        features,
        K,
        {
            "points": len(points),
            "registered_frames": len(poses),
            "selected_frames": len(frames),
            "reprojection_median_px": float(np.median(residuals)),
            "rejected_frames": rejected,
            "recovery_anchor_attempts": recovered_anchors,
            "intrinsics_source": camera.source
            if camera is not None
            else (
                "provided focal length"
                if config.focal_px
                else "heuristic; not calibrated"
            ),
            "scale": "arbitrary monocular",
        },
    )


def bundle_adjust(reconstruction, max_points=300, max_nfev=30):
    """Refine points with fixed camera poses; explicitly not full camera BA."""
    r = reconstruction
    count = min(len(r.points), max_points)
    obs = [
        (fid, k, pid)
        for (fid, k), pid in r.tracks.items()
        if pid < count and fid in r.poses
    ]

    def residual(x):
        p = x.reshape(-1, 3)
        return np.concatenate(
            [
                project(p[pid : pid + 1], r.intrinsics, r.poses[fid])[0][0]
                - r.features[fid][0][k]
                for fid, k, pid in obs
            ]
        )

    # Sparse Jacobian avoids a quadratic dense finite-difference cost.
    from scipy.sparse import lil_matrix

    sparsity = lil_matrix((2 * len(obs), 3 * count), dtype=int)
    for i, (_, _, pid) in enumerate(obs):
        sparsity[2 * i : 2 * i + 2, 3 * pid : 3 * pid + 3] = 1
    fit = least_squares(
        residual,
        r.points[:count].ravel(),
        jac_sparsity=sparsity,
        loss="soft_l1",
        max_nfev=max_nfev,
    )
    r.points[:count] = fit.x.reshape(-1, 3)
    r.diagnostics["point_refinement"] = {
        "points": count,
        "cost": float(fit.cost),
        "evaluations": fit.nfev,
        "converged": bool(fit.success),
    }
    return r
