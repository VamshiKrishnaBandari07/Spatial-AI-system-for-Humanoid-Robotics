"""Visibility requires measured depth, not holes in a sparse point cloud."""

import numpy as np
from typing import Any, Sequence
from numpy.typing import NDArray
from .geometry import project, Reconstruction


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


def crossview_free_surface(
    points: Sequence[Sequence[float]] | NDArray[np.float64],
    frame_id: str,
    reconstruction: Reconstruction,
    tolerance: float,
    min_angle: float = 2.0,
) -> dict[str, Any]:
    """Check observed background depth and textured patches in a second viewpoint.

    This is model-depth cross-validation, not measured free-space ground truth.
    Untextured patches, holes, insufficient parallax and inconsistent depth fail
    closed. It cannot infer absence merely because no detector box was returned.
    """
    import cv2

    current = reconstruction.surfaces[frame_id]
    pose = reconstruction.poses[frame_id]
    image = getattr(reconstruction, "features", {}).get(frame_id)
    if image is None:
        return {"supported_fraction": 0.0, "supporting_frames": []}
    image = image[2]
    uv, z = project(np.asarray(points), current["intrinsics"], pose)
    supported = 0
    supporting_frames = set()
    camera = -pose[:3, :3].T @ pose[:3, 3]
    for pixel, target_depth in zip(uv, z):
        if (
            not np.isfinite(pixel).all()
            or not np.isfinite(target_depth)
            or target_depth <= 0
        ):
            continue
        x, y = np.rint(pixel).astype(int)
        depth = current["depth"]
        h, w = depth.shape
        if not 3 <= x < w - 3 or not 3 <= y < h - 3:
            continue
        measured = float(depth[y, x])
        if not np.isfinite(measured) or measured <= target_depth + tolerance:
            continue
        content = current.get("valid_pixels", np.ones(depth.shape, bool))
        if not content[y, x]:
            continue
        confidence = current.get("confidence")
        if confidence is not None and confidence[y, x] < np.quantile(
            confidence[content], 0.5
        ):
            continue
        camera_point = (
            np.linalg.inv(current["intrinsics"]) @ np.array([x, y, 1.0]) * measured
        )
        world = (camera_point - pose[:3, 3]) @ pose[:3, :3]
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
        patch = gray[y - 2 : y + 3, x - 2 : x + 3].astype(float)
        if patch.std() < 5:
            continue
        ray = world - camera
        ray /= max(np.linalg.norm(ray), 1e-12)
        for peer_id, peer in reconstruction.surfaces.items():
            if peer_id == frame_id:
                continue
            other_image = getattr(reconstruction, "features", {}).get(peer_id)
            if other_image is None:
                continue
            other_pose = reconstruction.poses[peer_id]
            other_camera = -other_pose[:3, :3].T @ other_pose[:3, 3]
            other_ray = world - other_camera
            other_ray /= max(np.linalg.norm(other_ray), 1e-12)
            angle = float(np.degrees(np.arccos(np.clip(ray @ other_ray, -1, 1))))
            if angle < min_angle:
                continue
            projected, pz = project(world[None], peer["intrinsics"], other_pose)
            if not np.isfinite(projected).all() or pz[0] <= 0:
                continue
            px, py = np.rint(projected[0]).astype(int)
            ph, pw = peer["depth"].shape
            if not 3 <= px < pw - 3 or not 3 <= py < ph - 3:
                continue
            if not peer.get("valid_pixels", np.ones((ph, pw), bool))[py, px]:
                continue
            peer_confidence = peer.get("confidence")
            peer_content = peer.get("valid_pixels", np.ones((ph, pw), bool))
            if peer_confidence is not None and peer_confidence[py, px] < np.quantile(
                peer_confidence[peer_content], 0.5
            ):
                continue
            observed = float(peer["depth"][py, px])
            if not np.isfinite(observed) or abs(observed - pz[0]) > tolerance:
                continue
            other = other_image[2]
            other = (
                cv2.cvtColor(other, cv2.COLOR_BGR2GRAY) if other.ndim == 3 else other
            )
            other_patch = other[py - 2 : py + 3, px - 2 : px + 3].astype(float)
            centered = patch - patch.mean()
            other_centered = other_patch - other_patch.mean()
            length = np.linalg.norm(centered) * np.linalg.norm(other_centered)
            if (
                length <= 1e-6
                or float(np.sum(centered * other_centered) / length) < 0.9
            ):
                continue
            supported += 1
            supporting_frames.add(peer_id)
            break
    return {
        "supported_fraction": supported / max(1, len(points)),
        "supporting_frames": sorted(supporting_frames),
        "source": "Textured cross-view depth/reprojection agreement",
        "calibrated": False,
        "measured_depth": bool(
            getattr(reconstruction, "diagnostics", {}).get("sensor_geometry_used")
        ),
    }
