"""Joint sparse camera/landmark adjustment, accepted only on held-out pixels."""

from copy import deepcopy
from typing import Any
import cv2
import numpy as np
from scipy.optimize import least_squares
from scipy.sparse import lil_matrix
from .geometry import Reconstruction, project


def refine_cameras(reconstruction: Reconstruction, seed: int = 7,
                   max_points: int = 300, max_nfev: int = 40) -> Reconstruction:
    """Optimize 3D points and camera rvec/tvec with fixed K and first two poses.

    Each camera has six parameters; each point has three. Fixing two baseline
    cameras preserves the monocular scale gauge. Deterministic withheld 2D
    measurements are never optimized. All original inputs remain untouched
    unless held-out native-pixel support and median errors do not deteriorate.
    This is geometric cross-validation, not verified physical calibration.
    """
    audit: dict[str, Any] = {"accepted": False, "method": "joint Huber BA; fixed K and first two cameras"}
    frames = list(reconstruction.poses)
    if len(frames) < 3:
        reconstruction.diagnostics["joint_camera_refinement"] = {**audit, "reason": "fewer than three cameras"}
        return reconstruction
    rows = [(fid,key,pid) for (fid,key),pid in reconstruction.tracks.items()
            if fid in reconstruction.poses and pid < max_points]
    ids = sorted({pid for _,_,pid in rows})
    if len(rows) < 60 or len(ids) < 12:
        reconstruction.diagnostics["joint_camera_refinement"] = {**audit, "reason": "insufficient landmark observations"}
        return reconstruction
    points = reconstruction.points[ids].copy()
    index = {pid:i for i,pid in enumerate(ids)}
    cameras = {fid:i for i,fid in enumerate(frames[2:])}
    npoints = len(points)
    params = np.concatenate([np.r_[cv2.Rodrigues(reconstruction.poses[fid][:3,:3])[0].ravel(),
                                       reconstruction.poses[fid][:3,3]] for fid in frames[2:]])
    initial = np.r_[params, points.ravel()]
    offset = len(params)
    train, held = [], []
    rng = np.random.default_rng(seed)
    for fid in frames:
        group = [r for r in rows if r[0] == fid]
        order = rng.permutation(len(group))
        cut = max(1,len(group)//5)
        held.extend(group[i] for i in order[:cut])
        train.extend(group[i] for i in order[cut:])

    def unpack(values: np.ndarray) -> tuple[dict[str,np.ndarray], np.ndarray]:
        poses = {fid:pose.copy() for fid,pose in reconstruction.poses.items()}
        for fid,i in cameras.items():
            poses[fid][:3,:3] = cv2.Rodrigues(values[6*i:6*i+3])[0]
            poses[fid][:3,3] = values[6*i+3:6*i+6]
        return poses, values[offset:].reshape(npoints,3)

    def residual(values: np.ndarray, observations: list) -> np.ndarray:
        poses, xyz = unpack(values)
        result = np.empty((len(observations),2))
        for fid in frames:
            positions = [i for i,r in enumerate(observations) if r[0] == fid]
            if not positions:
                continue
            point_ids = [index[observations[i][2]] for i in positions]
            pixels = np.array([reconstruction.features[fid][0][observations[i][1]] for i in positions])
            result[positions] = project(xyz[point_ids], reconstruction.intrinsics, poses[fid])[0]-pixels
        return result.ravel()

    sparsity = lil_matrix((2*len(train),len(initial)),dtype=int)
    for i,(fid,_,pid) in enumerate(train):
        start = offset+3*index[pid]
        sparsity[2*i:2*i+2,start:start+3] = 1
        if fid in cameras:
            start = 6*cameras[fid]
            sparsity[2*i:2*i+2,start:start+6] = 1
    fit = least_squares(lambda x:residual(x,train),initial,jac_sparsity=sparsity,
                        loss="huber",f_scale=1.,max_nfev=max_nfev)
    original = np.linalg.norm(residual(initial,held).reshape(-1,2),axis=1)
    refined = np.linalg.norm(residual(fit.x,held).reshape(-1,2),axis=1)
    poses, xyz = unpack(fit.x)
    positive = all(np.all(project(xyz[[index[pid] for f,_,pid in rows if f==fid]],
                         reconstruction.intrinsics,poses[fid])[1] > 0) for fid in frames)
    rotations = [np.linalg.norm(cv2.Rodrigues(poses[fid][:3,:3] @ reconstruction.poses[fid][:3,:3].T)[0]) for fid in frames]
    bounded = max(rotations) < np.deg2rad(10)
    training_improved = np.linalg.norm(residual(fit.x,train)) < np.linalg.norm(residual(initial,train))
    accepted = bool(fit.success and np.isfinite(refined).all() and positive and bounded and training_improved
                    and np.median(refined) <= np.median(original)
                    and (refined < 2).sum() >= (original < 2).sum()
                    and np.quantile(refined,.95) <= np.quantile(original,.95))
    audit.update(accepted=accepted, heldout_observations=len(held),
                 original_heldout_median_px=float(np.median(original)),
                 refined_heldout_median_px=float(np.median(refined)),
                 original_heldout_inliers=int((original<2).sum()),
                 refined_heldout_inliers=int((refined<2).sum()),
                 solver_converged=bool(fit.success), evaluations=fit.nfev,
                 maximum_rotation_change_degrees=float(np.rad2deg(max(rotations))))
    result = deepcopy(reconstruction) if accepted else reconstruction
    if accepted:
        result.poses = poses
        result.points[ids] = xyz
    result.diagnostics["joint_camera_refinement"] = audit
    return result
