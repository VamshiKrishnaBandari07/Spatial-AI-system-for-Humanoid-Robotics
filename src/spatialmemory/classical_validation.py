"""Gated classical camera cross-validation of a neural reconstruction."""

from pathlib import Path
from typing import Sequence
import json
import cv2
import numpy as np

from .contracts import Config, Frame
from .geometry import Reconstruction, reconstruct, match, project
from .ingestion import keyframes
from .pixel_mapping import native_to_geometry_matrix
from .registration import ransac, transform


def validate_classical_cameras(
    neural: Reconstruction, frames: Sequence[Frame], output: Path, config: Config
) -> Reconstruction:
    """Correct camera rays only after correspondence and held-out Sim(3) checks.

    SIFT/SfM uses the inferred phone focal length, not claimed real calibration.
    The reference-frame neural point head supplies a scale gauge. All finite
    correspondences may compete; global confidence rank is NOT the acceptance
    criterion. RANSAC and withheld 3D residuals reject inconsistent geometry.
    Original point-head and depth-head estimates remain separate after correction.
    Failure is explicitly recorded, without an identity-alignment substitute.
    """
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    audit = {
        "accepted": False,
        "calibrated": False,
        "metric": False,
        "source": "SIFT essential/PnP + held-out Sim(3)",
        "static_within_recording_assumption": True,
    }

    def reject(reason):
        audit["reason"] = reason
        neural.diagnostics["classical_pose_validation"] = audit
        (output / "classical_camera_validation.json").write_text(
            json.dumps(audit, indent=2)
        )
        return neural

    if config.metric_geometry_enabled:
        return reject(
            "Measured sensor geometry is preserved; monocular camera correction is inapplicable"
        )
    if not neural.surfaces:
        return reject("Neural depth/point maps unavailable")
    lookup = {frame.id: frame for frame in frames}
    reference_id = next(iter(neural.poses))
    image = cv2.imread(lookup[reference_id].path)
    shape = neural.surfaces[reference_id]["depth"].shape
    affine = native_to_geometry_matrix(image.shape, shape, config.neural_preprocessing)
    native_k = np.linalg.inv(affine) @ neural.surfaces[reference_id]["intrinsics"]
    focal = float((native_k[0, 0] + native_k[1, 1]) / 2)
    classical_config = config.model_copy(
        update={"geometry_backend": "sparse", "focal_px": focal}
    )
    try:
        classical = reconstruct(keyframes(list(frames), 24), classical_config)
    except (ValueError, cv2.error) as exc:
        return reject(f"Classical reconstruction failed: {exc}")
    audit["classical_reconstruction"] = classical.diagnostics
    # Recover earlier/lost neural keyframes from the completed 3D landmark gallery.
    for fid in neural.poses:
        if fid in classical.poses:
            continue
        if fid not in classical.features:
            return reject(f"Neural keyframe {fid} is outside classical coverage")
        pixels, descriptor, _ = classical.features[fid]
        candidates = {}
        anchors = sorted(
            classical.poses,
            key=lambda key: abs(lookup[key].timestamp - lookup[fid].timestamp),
        )[:8]
        for anchor in anchors:
            for i, j in match(classical.features[anchor][1], descriptor):
                pid = classical.tracks.get((anchor, i))
                if pid is not None:
                    candidates.setdefault(j, set()).add(pid)
        pairs = [
            (key, next(iter(ids))) for key, ids in candidates.items() if len(ids) == 1
        ]
        if len(pairs) < 12:
            return reject(f"Insufficient relocalization evidence for {fid}")
        keys, pids = np.asarray(pairs).T
        order = np.random.default_rng(config.seed).permutation(len(pairs))
        held = order[: max(4, len(pairs) // 5)]
        train = order[len(held) :]
        ok, rv, tv, inliers = cv2.solvePnPRansac(
            classical.points[pids[train]],
            pixels[keys[train]],
            classical.intrinsics,
            None,
            iterationsCount=300,
            reprojectionError=config.reprojection_px,
            confidence=0.999,
        )
        if not ok or inliers is None or len(inliers) < 8:
            return reject(f"Relocalization PnP failed for {fid}")
        pose = np.eye(4)
        pose[:3, :3] = cv2.Rodrigues(rv)[0]
        pose[:3, 3] = tv[:, 0]
        projected, z = project(classical.points[pids[held]], classical.intrinsics, pose)
        errors = np.linalg.norm(projected - pixels[keys[held]], axis=1)
        if np.mean((errors <= config.reprojection_px) & (z > 0)) < 0.75:
            return reject(f"Held-out relocalization failed for {fid}")
        classical.poses[fid] = pose
        audit.setdefault("relocalized_frames", []).append(
            {
                "frame_id": fid,
                "heldout_points": len(held),
                "heldout_inliers": int(
                    ((errors <= config.reprojection_px) & (z > 0)).sum()
                ),
            }
        )
        for key, pid in pairs:
            classical.tracks[(fid, int(key))] = int(pid)
    # Reference frame has both real image correspondences and neural point predictions.
    surface = neural.surfaces[reference_id]
    affine = native_to_geometry_matrix(
        image.shape, surface["depth"].shape, config.neural_preprocessing
    )
    source = []
    target = []
    if config.classical_bundle_adjustment_enabled:
        from .camera_refinement import refine_cameras
        classical = refine_cameras(classical, config.seed)
        audit["joint_camera_refinement"] = classical.diagnostics.get("joint_camera_refinement")
    seen = set()
    for (fid, key), pid in classical.tracks.items():
        if fid != reference_id or pid in seen:
            continue
        uv = classical.features[fid][0][key]
        x, y = np.rint((affine @ np.r_[uv, 1])[:2]).astype(int)
        h, w = surface["depth"].shape
        if not 0 <= x < w or not 0 <= y < h or not surface["valid_pixels"][y, x]:
            continue
        predicted = surface["points"][y, x]
        if not np.isfinite(predicted).all():
            continue
        source.append(classical.points[pid])
        target.append(predicted)
        seen.add(pid)
    source = np.asarray(source).reshape(-1, 3)
    target = np.asarray(target).reshape(-1, 3)
    if len(source) < 12:
        return reject("Insufficient reference-frame 3D correspondences")
    order = np.random.default_rng(config.seed).permutation(len(source))
    held = order[: max(4, len(source) // 5)]
    train = order[len(held) :]
    candidate, _ = ransac(
        source[train], target[train], config.registration_threshold, config.seed
    )
    if not candidate.valid:
        return reject("Reference-frame Sim(3) fit rejected")
    errors = np.linalg.norm(
        transform(
            source[held],
            candidate.scale,
            np.asarray(candidate.rotation),
            np.asarray(candidate.translation),
        )
        - target[held],
        axis=1,
    )
    good = errors <= config.registration_threshold
    if good.mean() < 0.6:
        return reject("Held-out neural/classical 3D agreement rejected")
    if not np.isfinite(candidate.scale) or not 0.0001 < candidate.scale < 10000:
        return reject("Invalid relative reconstruction scale")
    rotation = np.asarray(candidate.rotation)
    translation = np.asarray(candidate.translation)
    scale = candidate.scale
    audit.update(
        accepted=True,
        relative_scale=scale,
        reference_frame=reference_id,
        fit_inliers=candidate.inliers,
        fit_correspondences=candidate.correspondences,
        fit_rmse=candidate.rmse,
        heldout_inliers=int(good.sum()),
        heldout_points=len(held),
        heldout_inlier_rmse=float(np.sqrt(np.mean(errors[good] ** 2))),
        intrinsics_source="VGGT-derived constant focal estimate; not measured calibration",
    )
    for fid, old_pose in list(neural.poses.items()):
        sfm_pose = classical.poses[fid]
        new_pose = np.eye(4)
        new_pose[:3, :3] = sfm_pose[:3, :3] @ rotation.T
        new_pose[:3, 3] = scale * sfm_pose[:3, 3] - new_pose[:3, :3] @ translation
        current = neural.surfaces[fid]
        native = cv2.imread(lookup[fid].path)
        affine = native_to_geometry_matrix(
            native.shape, current["depth"].shape, config.neural_preprocessing
        )
        k = affine @ classical.intrinsics
        y, x = np.indices(current["depth"].shape)
        rays = np.stack([x, y, np.ones_like(x)], axis=-1) @ np.linalg.inv(k).T
        old_camera = current["points"] @ old_pose[:3, :3].T + old_pose[:3, 3]
        # Preserve independently predicted point-head z, do not replace it with depth-head z.
        corrected_camera = rays * old_camera[..., 2, None]
        current["points"] = (corrected_camera - new_pose[:3, 3]) @ new_pose[:3, :3]
        current["intrinsics"] = k
        current["camera_source"] = (
            "Classical SIFT/PnP, aligned with held-out neural geometry checks"
        )
        neural.poses[fid] = new_pose
        for (frame, key), pid in neural.tracks.items():
            if frame == fid:
                px, py = np.rint(neural.features[fid][0][key]).astype(int)
                neural.points[pid] = current["points"][py, px]
    neural.intrinsics = next(iter(neural.surfaces.values()))["intrinsics"]
    neural.diagnostics["classical_pose_validation"] = audit
    (output / "classical_camera_validation.json").write_text(
        json.dumps(audit, indent=2)
    )
    return neural
