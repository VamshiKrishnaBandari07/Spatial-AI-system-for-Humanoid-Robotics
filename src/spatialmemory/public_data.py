"""Dataset-only native RGB-D preparation; models and inference are unchanged."""

import hashlib
import json
import math
from pathlib import Path
import shutil
import urllib.request
import zipfile

import cv2
import numpy as np
from pydantic import Field
from scipy.spatial.transform import Rotation

from .datasets import encode_images
from .inputs import (
    Camera,
    DepthFrame,
    DepthMetadata,
    Pose,
    Poses,
    Recording,
    Strict,
    read_depth,
)


class SequenceFrame(Strict):
    rgb: str
    depth: str
    timestamp_ns: int = Field(ge=0)
    source_frame: str
    camera: Camera
    T_world_camera: list[list[float]]
    depth_unit: str = "millimetres"
    invalid_depth: int = 0
    synchronization_error_seconds: float = Field(ge=0)
    rgb_depth_aligned: bool = True
    calibration_kind: str = "rgb"
    native_T_world_camera: list[list[float]] | None = None
    rotation_projection_frobenius: float = 0


class DatasetSequence(Strict):
    schema_version: str = "1.0"
    dataset: str
    scene_id: str
    recording_id: str
    split: str
    native_split: str | None = None
    source: str
    license: str
    timestamp_source: str
    pose_convention: str = "T_world_camera"
    coordinate_convention: str = "opencv: X right, Y down, Z forward"
    translation_unit: str = "metres"
    change_annotations: None = None
    frames: list[SequenceFrame]


def valid_transform(matrix):
    matrix = np.asarray(matrix, dtype=float)
    if (
        matrix.shape != (4, 4)
        or not np.isfinite(matrix).all()
        or not np.allclose(matrix[3], [0, 0, 0, 1])
    ):
        raise ValueError("Invalid homogeneous camera transform")
    rotation = matrix[:3, :3]
    if not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-4) or not np.isclose(
        np.linalg.det(rotation), 1, atol=1e-4
    ):
        raise ValueError("Camera rotation is not a proper orthonormal matrix")
    return matrix


def load_frame(frame):
    """Use the existing calibrated camera/depth readers; return CHW RGB and metric depth."""
    rgb = cv2.imread(frame.rgb)
    if rgb is None or rgb.shape[:2] != (frame.camera.height, frame.camera.width):
        raise ValueError("RGB dimensions disagree with source calibration")
    meta = DepthMetadata(
        unit=frame.depth_unit,
        width=frame.camera.width,
        height=frame.camera.height,
        invalid_value=frame.invalid_depth,
        camera="per-frame source calibration",
        rgb_aligned=True,
        scale_verified=True,
        frames=[],
    )
    depth = read_depth(frame.depth, meta)
    depth[depth <= 0] = np.nan
    pose = valid_transform(frame.T_world_camera)
    K = frame.camera.matrix(frame.camera.width, frame.camera.height)
    return {
        "rgb": cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
        .transpose(2, 0, 1)
        .astype(np.float32)
        / 255,
        "depth_metres": depth.astype(np.float32),
        "K": K,
        "K_rgb": K if frame.calibration_kind == "rgb" else None,
        "rgb_depth_aligned": frame.rgb_depth_aligned,
        "T_world_camera": pose,
        "timestamp_ns": frame.timestamp_ns,
    }


def arkit_sequence(root, scene_id, recording_id, split, max_frames=48):
    root = Path(root).resolve()
    trajectory = np.loadtxt(root / "lowres_wide.traj", ndmin=2)
    if trajectory.shape[1] != 7 or not np.isfinite(trajectory).all():
        raise ValueError(
            "ARKit trajectory requires finite timestamp, rotation-vector and translation"
        )
    if np.any(np.diff(trajectory[:, 0]) <= 0):
        raise ValueError("ARKit trajectory timestamps must increase")
    calibrations = sorted((root / "lowres_wide_intrinsics").glob("*.pincam"))
    if not calibrations:
        raise ValueError("ARKit calibration is unavailable")
    calibration_times = np.array(
        [float(p.stem.rsplit("_", 1)[1]) for p in calibrations]
    )
    images = sorted((root / "lowres_wide").glob("*.png"))
    if not images:
        raise ValueError("ARKit RGB frames unavailable")
    images = sorted(images, key=lambda p: float(p.stem.rsplit("_", 1)[1]))
    if max_frames < 2:
        raise ValueError("At least two multi-view frames required")
    # Match native timestamps before budgeting, so sparse pose sampling does not
    # accidentally discard most of an otherwise valid RGB-D sequence.
    eligible = []
    for index, rgb in enumerate(images):
        time = float(rgb.stem.rsplit("_", 1)[1])
        if (
            (root / "lowres_depth" / rgb.name).exists()
            and np.min(abs(trajectory[:, 0] - time)) <= 0.005
            and np.min(abs(calibration_times - time)) <= 0.0011
        ):
            eligible.append(index)
    if len(eligible) < 2:
        raise ValueError("Fewer than two synchronized calibrated ARKit RGB-D frames")
    chosen = np.array(eligible)[
        np.unique(
            np.linspace(0, len(eligible) - 1, min(max_frames, len(eligible)))
            .round()
            .astype(int)
        )
    ]
    frames = []
    for index in chosen:
        rgb = images[index]
        time = float(rgb.stem.rsplit("_", 1)[1])
        depth = root / "lowres_depth" / rgb.name
        if not depth.exists():
            continue
        pose_index = int(np.argmin(abs(trajectory[:, 0] - time)))
        calibration_index = int(np.argmin(abs(calibration_times - time)))
        delta = abs(trajectory[pose_index, 0] - time)
        if delta > 0.005 or abs(calibration_times[calibration_index] - time) > 0.0011:
            continue
        w, h, fx, fy, cx, cy = np.loadtxt(calibrations[calibration_index])
        camera = Camera(
            width=int(w),
            height=int(h),
            fx=fx,
            fy=fy,
            cx=cx,
            cy=cy,
            source="ARKitScenes native per-frame .pincam",
        )
        # Apple's official TenFpsDataLoader inverts world-to-camera Rodrigues extrinsics.
        world_to_camera = np.eye(4)
        world_to_camera[:3, :3] = Rotation.from_rotvec(
            trajectory[pose_index, 1:4]
        ).as_matrix()
        world_to_camera[:3, 3] = trajectory[pose_index, 4:7]
        frame = SequenceFrame(
            rgb=str(rgb),
            depth=str(depth),
            timestamp_ns=round(time * 1e9),
            source_frame=rgb.stem,
            camera=camera,
            T_world_camera=np.linalg.inv(world_to_camera).tolist(),
            synchronization_error_seconds=float(delta),
        )
        load_frame(frame)
        frames.append(frame)
    if len(frames) < 2:
        raise ValueError("Fewer than two synchronized calibrated ARKit RGB-D frames")
    return DatasetSequence(
        dataset="ARKitScenes",
        scene_id=scene_id,
        recording_id=recording_id,
        split=split,
        native_split=next(
            (p.name for p in root.parents if p.name in ["Training", "Validation"]), None
        ),
        source="https://github.com/apple/ARKitScenes",
        license="Apple ARKitScenes LICENSE; retain source license",
        timestamp_source="Native filename/traj seconds; nearest pose <=5ms, intrinsics <=1.1ms",
        frames=frames,
    )


def seven_scenes_sequence(root, scene_id, recording_id, split, camera, max_frames=48):
    """Calibration must be supplied from verified dataset documentation; never guessed."""
    root = Path(root).resolve()
    images = sorted(root.glob("frame-*.color.png"))
    if not images or max_frames < 2:
        raise ValueError("7-Scenes RGB frames unavailable or invalid frame budget")
    indices = np.unique(
        np.linspace(0, len(images) - 1, min(max_frames, len(images)))
        .round()
        .astype(int)
    )
    frames = []
    for index in indices:
        rgb = images[index]
        stem = rgb.name.removesuffix(".color.png")
        depth = root / (stem + ".depth.png")
        pose = root / (stem + ".pose.txt")
        if not depth.exists() or not pose.exists():
            continue
        native_index = int(stem.split("-")[-1])
        native_pose = np.loadtxt(pose)
        if (
            native_pose.shape != (4, 4)
            or not np.isfinite(native_pose).all()
            or not np.allclose(native_pose[3], [0, 0, 0, 1])
        ):
            raise ValueError("Invalid native 7-Scenes homogeneous pose")
        rotation = native_pose[:3, :3]
        if (
            np.max(abs(rotation.T @ rotation - np.eye(3))) > 0.001
            or abs(np.linalg.det(rotation) - 1) > 0.002
        ):
            raise ValueError(
                "Native 7-Scenes pose rotation drift exceeds supported tolerance"
            )
        u, _, vt = np.linalg.svd(rotation)
        corrected = native_pose.copy()
        corrected[:3, :3] = u @ vt
        correction = float(np.linalg.norm(corrected[:3, :3] - rotation))
        frame = SequenceFrame(
            rgb=str(rgb),
            depth=str(depth),
            timestamp_ns=round(native_index / 30 * 1e9),
            source_frame=stem,
            camera=camera,
            T_world_camera=valid_transform(corrected).tolist(),
            native_T_world_camera=native_pose.tolist(),
            rotation_projection_frobenius=correction,
            invalid_depth=65535,
            rgb_depth_aligned=False,
            calibration_kind="depth_nominal",
            synchronization_error_seconds=0,
        )
        load_frame(frame)
        frames.append(frame)
    if len(frames) < 2:
        raise ValueError("Fewer than two valid 7-Scenes RGB-D/pose frames")
    return DatasetSequence(
        dataset="7-Scenes",
        scene_id=scene_id,
        recording_id=recording_id,
        split=split,
        source="https://www.microsoft.com/en-us/research/project/rgb-d-dataset-7-scenes/",
        license="Microsoft Research License Agreement (MSR-LA), noncommercial research",
        timestamp_source="Derived source frame-index / nominal 30Hz; hardware timestamps unavailable",
        frames=frames,
    )


def export_recording(sequence, output):
    """Single recording, not an invented before/after scene. Keep all source frame calibration."""
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    (output / "depth").mkdir()
    images = [Path(f.rgb) for f in sequence.frames]
    encode_images(images, output / "rgb.mp4", fps=30)
    # The video is a model-input carrier; source timestamps override its synthetic encoding clock.
    times = [f.timestamp_ns / 1e9 for f in sequence.frames]
    (output / "timestamps.json").write_text(json.dumps(times))
    source_cameras = [f.camera.model_dump() for f in sequence.frames]
    (output / "per_frame_intrinsics.json").write_text(
        json.dumps(source_cameras, indent=2)
    )
    first = sequence.frames[0].camera
    K = first.matrix(first.width, first.height)
    constant = all(
        np.allclose(K, f.camera.matrix(first.width, first.height), rtol=1e-6, atol=1e-5)
        for f in sequence.frames
    )
    camera_path = output / "camera.json" if constant else None
    if sequence.dataset == "7-Scenes":
        # Microsoft explicitly states raw RGB/depth are uncalibrated. The
        # nominal KinectFusion depth intrinsics must not become RGB calibration.
        (output / "depth_camera_nominal.json").write_text(
            first.model_dump_json(indent=2)
        )
        camera_path = None
    if camera_path:
        camera_path.write_text(first.model_dump_json(indent=2))
    poses = []
    depths = []
    for index, frame in enumerate(sequence.frames):
        data = load_frame(frame)
        depth = np.nan_to_num(data["depth_metres"], nan=0)
        name = f"{index:06d}.npy"
        np.save(output / "depth" / name, depth)
        depths.append(
            DepthFrame(
                file=name, timestamp_ns=frame.timestamp_ns, rgb_frame_index=index
            )
        )
        pose = data["T_world_camera"]
        poses.append(
            Pose(
                timestamp_ns=frame.timestamp_ns,
                translation=pose[:3, 3].tolist(),
                quaternion_xyzw=Rotation.from_matrix(pose[:3, :3]).as_quat().tolist(),
            )
        )
    pose_path = output / "poses.json"
    pose_path.write_text(
        Poses(
            translation_unit="metres",
            source="Native dataset poses, explicit camera-to-world",
            poses=poses,
        ).model_dump_json(indent=2)
    )
    depth_path = None
    if constant and all(f.rgb_depth_aligned for f in sequence.frames):
        depth_path = output / "depth"
        metadata = DepthMetadata(
            unit="metres",
            width=first.width,
            height=first.height,
            camera=str(camera_path),
            rgb_aligned=True,
            scale_verified=True,
            frames=depths,
        )
        (depth_path / "metadata.json").write_text(metadata.model_dump_json(indent=2))
    recording = Recording(
        video=str(output / "rgb.mp4"),
        camera=str(camera_path) if camera_path else None,
        depth_dir=str(depth_path) if depth_path else None,
        poses=str(pose_path),
        timestamps=str(output / "timestamps.json"),
    )
    (output / "sequence.json").write_text(sequence.model_dump_json(indent=2))
    (output / "recording.json").write_text(recording.model_dump_json(indent=2))
    (output / "conversion.json").write_text(
        json.dumps(
            {
                "constant_intrinsics": constant,
                "variable_intrinsics": "Preserved per-frame; current core accepts one camera. Variable-calibration depth is not advertised for fusion.",
                "annotations": None,
                "pair": None,
                "video_clock": "derived 30fps; source timestamp sidecar retained",
            },
            indent=2,
        )
    )
    return recording


def write_splits(sequences, output):
    groups = {}
    result = {s: [] for s in ["train", "validation", "test"]}
    for path in sequences:
        sequence = DatasetSequence.model_validate_json(Path(path).read_text())
        if sequence.split not in result:
            raise ValueError("Split must be train, validation or test")
        key = (sequence.dataset.removeprefix("synthetic_"), sequence.scene_id)
        if key in groups and groups[key] != sequence.split:
            raise ValueError("Scene identity leaks across splits")
        groups[key] = sequence.split
        result[sequence.split].append(
            {
                "sequence_manifest": str(Path(path).resolve()),
                "dataset": sequence.dataset,
                "scene_id": sequence.scene_id,
                "recording_id": sequence.recording_id,
                "native_split": sequence.native_split,
            }
        )
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    for split, entries in result.items():
        path = output / (split + ".json")
        if path.exists():
            raise FileExistsError(path)
        path.write_text(
            json.dumps(
                {
                    "schema_version": "1.0",
                    "split": split,
                    "recordings": entries,
                    "policy": "Scene-disjoint; native splits preserved; empty means no verified recordings",
                },
                indent=2,
            )
        )
    return result


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download_asset(url, target, max_bytes=512 * 1024 * 1024):
    """Bounded HTTPS acquisition, local integrity journal, no overwrites or unverified reuse."""
    if not url.startswith("https://"):
        raise ValueError("Use the official HTTPS source")
    target = Path(target)
    journal = target.with_suffix(target.suffix + ".download.json")
    if target.exists():
        if (
            not journal.exists()
            or sha256(target) != json.loads(journal.read_text())["sha256"]
        ):
            raise ValueError(
                "Existing asset has no matching integrity journal; preserve and inspect it"
            )
        return target
    target.parent.mkdir(parents=True, exist_ok=True)
    if shutil.disk_usage(target.parent).free < max_bytes + 1024**3:
        raise ValueError("Insufficient disk reserve for download")
    temporary = target.with_suffix(target.suffix + ".partial")
    try:
        with urllib.request.urlopen(url, timeout=60) as response:
            size = response.headers.get("Content-Length")
            if size and int(size) > max_bytes:
                raise ValueError("Official asset exceeds download budget")
            count = 0
            with temporary.open("xb") as file:
                while chunk := response.read(1024 * 1024):
                    count += len(chunk)
                    if count > max_bytes:
                        raise ValueError("Transfer budget exceeded")
                    file.write(chunk)
            if size and count != int(size):
                raise ValueError("Truncated download")
        checksum = sha256(temporary)
        temporary.replace(target)
        journal.write_text(
            json.dumps(
                {
                    "source": url,
                    "bytes": count,
                    "sha256": checksum,
                    "official_checksum_available": False,
                },
                indent=2,
            )
        )
        return target
    except Exception:
        # Preserve interrupted partial files for inspection; never overwrite them.
        raise


def extract_zip(archive, output, max_bytes=2 * 1024**3, exclude_names=()):
    output = Path(output).resolve()
    with zipfile.ZipFile(archive) as zipped:
        members = zipped.infolist()
        if sum(m.file_size for m in members) > max_bytes:
            raise ValueError("Extraction budget exceeded")
        for member in members:
            if (
                not (output / member.filename).resolve().is_relative_to(output)
                or (member.external_attr >> 16) & 0o170000 == 0o120000
            ):
                raise ValueError("Unsafe archive path or symlink")
        output.mkdir(parents=True, exist_ok=False)
        for member in members:
            if Path(member.filename).name in exclude_names:
                continue
            target = output / member.filename
            if member.is_dir():
                target.mkdir(parents=True, exist_ok=True)
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            with zipped.open(member) as source, target.open("xb") as dest:
                shutil.copyfileobj(source, dest)
    return output


def replica_atlas(path):
    """Official Replica PTexLib: square raw little-endian RGB half floats, not Radiance HDR."""
    size = Path(path).stat().st_size
    dimension = math.isqrt(size // 6)
    if size != dimension * dimension * 6 or dimension == 0:
        raise ValueError("Invalid native Replica atlas size")
    data = np.memmap(path, mode="r", dtype="<f2", shape=(dimension, dimension, 3))
    if not np.isfinite(data).all():
        raise ValueError("Nonfinite native HDR texture")
    return data


def download_zip_member(url, member_name, target, max_bytes=512 * 1024**2):
    """Fetch a selected official archive member with validated HTTP byte ranges."""
    import io

    class Ranges(io.RawIOBase):
        def __init__(self):
            with urllib.request.urlopen(
                urllib.request.Request(url, method="HEAD"), timeout=30
            ) as response:
                self.length = int(response.headers["Content-Length"])
            self.position = 0
            self.transferred = 0

        def seekable(self):
            return True

        def readable(self):
            return True

        def tell(self):
            return self.position

        def seek(self, offset, whence=0):
            position = offset + (
                0 if whence == 0 else self.position if whence == 1 else self.length
            )
            if position < 0:
                raise ValueError("Invalid negative archive offset")
            self.position = position
            return position

        def read(self, size=-1):
            size = min(
                self.length - self.position,
                size if size >= 0 else self.length - self.position,
            )
            if size <= 0:
                return b""
            if self.transferred + size > max_bytes + 4 * 1024**2:
                raise ValueError("Archive transfer budget exceeded")
            end = self.position + size - 1
            req = urllib.request.Request(
                url, headers={"Range": f"bytes={self.position}-{end}"}
            )
            with urllib.request.urlopen(req, timeout=60) as response:
                expected = f"bytes {self.position}-{end}/{self.length}"
                if (
                    response.status != 206
                    or response.headers.get("Content-Range") != expected
                ):
                    raise ValueError("Official source did not honor exact byte range")
                data = response.read(size + 1)
            if len(data) != size:
                raise ValueError("Truncated remote archive range")
            self.position += size
            self.transferred += size
            return data

    target = Path(target)
    journal = target.with_suffix(target.suffix + ".download.json")
    if target.exists():
        if (
            not journal.exists()
            or sha256(target) != json.loads(journal.read_text())["sha256"]
        ):
            raise ValueError("Existing file integrity journal mismatch; preserved")
        return target
    target.parent.mkdir(parents=True, exist_ok=True)
    if shutil.disk_usage(target.parent).free < max_bytes + 1024**3:
        raise ValueError("Insufficient disk reserve")
    remote = Ranges()
    with zipfile.ZipFile(remote) as archive:
        names = archive.namelist()
        print("Official archive members:", names, flush=True)
        if member_name is None:
            return names
        entry = archive.getinfo(member_name)
        if entry.file_size > max_bytes:
            raise ValueError("Selected member exceeds budget")
        temporary = target.with_suffix(target.suffix + ".partial")
        count = 0
        with archive.open(entry) as source, temporary.open("xb") as file:
            while chunk := source.read(8 * 1024**2):
                count += len(chunk)
                if count > max_bytes:
                    raise ValueError("Expanded member budget exceeded")
                file.write(chunk)
        if count != entry.file_size:
            raise ValueError("Truncated archive member")
        temporary.replace(target)
    journal.write_text(
        json.dumps(
            {
                "source": url,
                "member": member_name,
                "bytes": count,
                "transfer_bytes": remote.transferred,
                "sha256": sha256(target),
                "verification": "HTTP exact range checks, ZIP member CRC, local SHA256",
                "official_checksum_available": False,
            },
            indent=2,
        )
    )
    return target


def verify_extracted_zip(archive, output, exclude_names=()):
    """Verify an existing extraction against original member CRCs; never overwrite data."""
    import zlib

    root = Path(output).resolve()
    verified = 0
    with zipfile.ZipFile(archive) as zipped:
        for member in zipped.infolist():
            if member.is_dir() or Path(member.filename).name in exclude_names:
                continue
            path = (root / member.filename).resolve()
            if not path.is_relative_to(root) or not path.is_file():
                raise ValueError("Missing or unsafe extracted file: " + member.filename)
            if path.stat().st_size != member.file_size:
                raise ValueError("Extracted size mismatch: " + member.filename)
            crc = 0
            with path.open("rb") as file:
                for chunk in iter(lambda: file.read(1024 * 1024), b""):
                    crc = zlib.crc32(chunk, crc)
            if crc != member.CRC:
                raise ValueError("Extracted CRC mismatch: " + member.filename)
            verified += 1
    return verified
