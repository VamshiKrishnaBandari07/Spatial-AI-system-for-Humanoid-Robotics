"""Calibrated depth/pose reconstruction, preserving missing pixels and sensor provenance."""

from pathlib import Path
import numpy as np

from .geometry import Reconstruction, extract
from .inputs import Camera, read_depth
from .public_data import valid_transform
from .rgbd_validation import unproject, transform_points, project


def rasterize_depth(depth, K_depth, K_rgb, T_rgb_depth, height, width):
    points, _ = unproject(depth, K_depth)
    points = transform_points(points, T_rgb_depth)
    points = points[points[:, 2] > 0]
    uv = np.rint(project(points, K_rgb)).astype(int)
    inside = (
        (uv[:, 0] >= 0) & (uv[:, 0] < width) & (uv[:, 1] >= 0) & (uv[:, 1] < height)
    )
    points, uv = points[inside], uv[inside]
    result = np.full((height, width), np.inf)
    np.minimum.at(result, (uv[:, 1], uv[:, 0]), points[:, 2])
    result[~np.isfinite(result)] = np.nan
    return result


def reconstruct_sensor(frames, files, neural_diagnostics=None):
    required = ["camera_data", "depth_metadata", "pose_data", "depth_dir"]
    if any(files.get(k) is None for k in required):
        raise ValueError(
            "Metric reconstruction requires calibration, depth metadata and known poses"
        )
    metadata, camera, poses = (
        files["depth_metadata"],
        files["camera_data"],
        files["pose_data"],
    )
    if (
        not metadata.scale_verified
        or metadata.unit == "relative"
        or poses.translation_unit != "metres"
    ):
        raise ValueError("Verified metric depth and metre-valued poses required")
    if camera.distortion_model != "none":
        raise ValueError("Rectify calibrated RGB before metric reconstruction")
    depth_camera = Camera.model_validate_json(Path(metadata.camera).read_text())
    Kd = depth_camera.matrix(metadata.width, metadata.height)
    T = np.eye(4) if metadata.rgb_aligned else valid_transform(metadata.T_rgb_depth)
    lookup = {f.rgb_frame_index: f for f in metadata.frames}
    features = extract(frames)
    points, colors, tracks, world_to_camera, surfaces = [], [], {}, {}, {}
    times = np.array([p.timestamp_ns for p in poses.poses])
    for frame in frames:
        entry = lookup.get(frame.index)
        if entry is None:
            raise ValueError("Selected RGB frame has no corresponding depth")
        j = int(np.argmin(abs(times - entry.timestamp_ns)))
        if abs(times[j] - entry.timestamp_ns) > 50_000_000:
            raise ValueError("Pose/depth timestamp mismatch")
        pose = poses.poses[j]
        from scipy.spatial.transform import Rotation

        Twc = np.eye(4)
        Twc[:3, :3] = Rotation.from_quat(pose.quaternion_xyzw).as_matrix()
        Twc[:3, 3] = pose.translation
        Twc = valid_transform(Twc)
        image = features[frame.id][2]
        h, w = image.shape[:2]
        K = camera.matrix(w, h)
        native = read_depth(Path(files["depth_dir"]) / entry.file, metadata)
        depth = rasterize_depth(native, Kd, K, T, h, w)
        yy, xx = np.mgrid[:h, :w]
        rays = np.stack(
            [(xx - K[0, 2]) / K[0, 0], (yy - K[1, 2]) / K[1, 1], np.ones((h, w))],
            axis=-1,
        )
        local = rays * depth[:, :, None]
        world = local @ Twc[:3, :3].T + Twc[:3, 3]
        valid = np.isfinite(world).all(axis=-1)
        sample = valid & (xx % 3 == 0) & (yy % 3 == 0)
        points.extend(world[sample])
        colors.extend(image[sample][:, ::-1] / 255)
        world_to_camera[frame.id] = np.linalg.inv(Twc)
        surfaces[frame.id] = {
            "points": world,
            "depth": depth,
            "confidence": valid.astype(float),
            "intrinsics": K,
            "depth_source": "Measured native depth projected with explicit RGB/depth calibration; holes remain unknown",
        }
    if len(points) < 30:
        raise ValueError("Insufficient metric surface samples")
    return Reconstruction(
        np.asarray(points),
        np.asarray(colors),
        world_to_camera,
        tracks,
        features,
        K,
        {
            "backend": "calibrated RGB-D",
            "points": len(points),
            "registered_frames": len(frames),
            "scale": "verified sensor metres",
            "neural_cross_validation": neural_diagnostics,
            "sensor_geometry_used": True,
            "depth_source": "native measured depth",
            "pose_convention": "T_world_camera; inverse stored for projection",
        },
        surfaces,
    )
