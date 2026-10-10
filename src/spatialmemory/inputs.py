"""Canonical input v1.0. Relative paths resolve against an explicit project root."""

import csv, json
from datetime import datetime
from pathlib import Path
from typing import Literal
import numpy as np
import cv2
from pydantic import BaseModel, Field, ConfigDict, model_validator
from .contracts import Change


class Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


class Camera(Strict):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)
    model: Literal["PINHOLE"] = "PINHOLE"
    width: int = Field(gt=0)
    height: int = Field(gt=0)
    fx: float = Field(gt=0)
    fy: float = Field(gt=0)
    cx: float
    cy: float
    distortion_model: Literal["none", "opencv"] = "none"
    distortion_coefficients: list[float] = Field(default_factory=list)
    source: str

    @model_validator(mode="after")
    def valid(self):
        if self.distortion_model == "none" and self.distortion_coefficients:
            raise ValueError("Distortion coefficients supplied for none model")
        if self.distortion_model == "opencv" and len(
            self.distortion_coefficients
        ) not in [4, 5, 8, 12, 14]:
            raise ValueError("Unsupported OpenCV distortion length")
        return self

    def matrix(self, width, height):
        sx = width / self.width
        sy = height / self.height
        return np.array(
            [
                [self.fx * sx, 0, self.cx * sx],
                [0, self.fy * sy, self.cy * sy],
                [0, 0, 1.0],
            ]
        )


class Recording(Strict):
    video: str
    camera: str | None = None
    depth_dir: str | None = None
    poses: str | None = None
    imu: str | None = None
    timestamps: str | None = None


class Manifest(Strict):
    schema_version: Literal["1.0"] = "1.0"
    scene_id: str = Field(min_length=1, pattern=r"^[A-Za-z0-9_.-]+$")
    environment: str = "indoor"
    coordinate_convention: Literal["opencv"] = "opencv"
    before: Recording
    after: Recording
    annotations: str | None = None
    rgb_only: bool = True
    dataset: dict | None = None


class AnnotationObservation(Strict):
    recording_id: str = Field(min_length=1)
    frame_id: str | None = None
    timestamp_ns: int | None = Field(default=None, ge=0)
    bbox_xyxy: list[float] | None = None
    mask: str | None = None
    source: str = Field(min_length=1)

    @model_validator(mode="after")
    def box(self):
        b = self.bbox_xyxy
        if b is not None and (
            len(b) != 4
            or not np.isfinite(b).all()
            or min(b) < 0
            or b[2] <= b[0]
            or b[3] <= b[1]
        ):
            raise ValueError(
                "Bounding box must be finite nonnegative xyxy with positive area"
            )
        if (b is not None or self.mask is not None) and self.frame_id is None:
            raise ValueError("Boxes and masks require a frame identifier")
        return self


class Annotation(Strict):
    object_id: str = Field(min_length=1)
    label: str = Field(min_length=1)
    change_type: Change
    before_position: list[float] | None = None
    after_position: list[float] | None = None
    visibility: Literal["OBSERVED", "OCCLUDED", "UNKNOWN"] = "UNKNOWN"
    annotation_source: str = Field(min_length=1)
    before_observation: AnnotationObservation | None = None
    after_observation: AnnotationObservation | None = None
    annotation_quality: Literal[
        "UNREVIEWED", "SINGLE_REVIEW", "INDEPENDENTLY_REVIEWED"
    ] = "UNREVIEWED"
    generation_method: (
        Literal[
            "human",
            "dataset_native",
            "model_suggestion_verified",
            "model_generated",
            "synthetic",
        ]
        | None
    ) = None
    annotator: str | None = None
    independent_reviewer: str | None = None
    reviewed_at: str | None = None
    native_object_id_before: str | None = None
    native_object_id_after: str | None = None
    annotation_level: Literal[1, 2, 3, 4] = 1
    before_geometry: str | None = None
    after_geometry: str | None = None
    measured_displacement: list[float] | None = None

    @model_validator(mode="after")
    def positions(self):
        for p in [
            self.before_position,
            self.after_position,
            self.measured_displacement,
        ]:
            if p is not None and (len(p) != 3 or not np.isfinite(p).all()):
                raise ValueError("Positions must be finite 3-vectors")
        if self.annotation_quality == "INDEPENDENTLY_REVIEWED" and (
            not self.annotator
            or not self.independent_reviewer
            or self.annotator.strip().casefold()
            == self.independent_reviewer.strip().casefold()
            or not self.reviewed_at
        ):
            raise ValueError(
                "Independent review requires distinct annotator/reviewer and review timestamp"
            )
        if self.annotation_quality == "SINGLE_REVIEW" and not self.annotator:
            raise ValueError("Reviewed annotations require an annotator")
        if self.reviewed_at is not None:
            timestamp = datetime.fromisoformat(self.reviewed_at.replace("Z", "+00:00"))
            if timestamp.tzinfo is None:
                raise ValueError("Review timestamp requires a timezone")
        observations = [
            o for o in [self.before_observation, self.after_observation] if o
        ]
        if self.annotation_level == 2 and not any(
            o.bbox_xyxy is not None or o.mask for o in observations
        ):
            raise ValueError("Level 2 annotations require frame-linked boxes or masks")
        if (
            self.annotation_level >= 3
            and self.before_position is None
            and self.after_position is None
        ):
            raise ValueError("Level 3+ annotations require a measured 3D position")
        if self.annotation_level == 4 and not (
            self.before_geometry or self.after_geometry
        ):
            raise ValueError("Level 4 annotations require a geometry asset reference")
        if (
            self.annotation_level == 4
            and self.change_type == Change.MOVED
            and self.measured_displacement is None
        ):
            raise ValueError("Level 4 movement requires a measured displacement")
        if self.change_type == Change.ADDED and (
            self.before_position is not None or self.before_observation is not None
        ):
            raise ValueError(
                "ADDED annotation cannot include an observed pre-existing object"
            )
        if self.change_type == Change.REMOVED and (
            self.after_position is not None or self.after_observation is not None
        ):
            raise ValueError(
                "REMOVED annotation cannot include an observed surviving object"
            )
        return self


class Annotations(Strict):
    schema_version: Literal["1.0"] = "1.0"
    scene_id: str
    coordinate_frame: str = "shared_world"
    position_unit: Literal[
        "metres", "millimetres", "reconstruction_units", "unknown"
    ] = "unknown"
    scale_verified: bool = False
    changes: list[Annotation]
    data_origin: Literal["REAL", "SYNTHETIC", "UNKNOWN"] = "UNKNOWN"
    exhaustive_object_inventory: bool = False
    provenance: str | None = None

    @model_validator(mode="after")
    def scale(self):
        if not self.scale_verified and self.position_unit in ["metres", "millimetres"]:
            raise ValueError("Metric annotations require verified scale")
        if len({a.object_id for a in self.changes}) != len(self.changes):
            raise ValueError("Duplicate object IDs")
        return self


class Pose(Strict):
    timestamp_ns: int = Field(ge=0)
    translation: list[float]
    quaternion_xyzw: list[float] | None = None
    rotation_matrix: list[list[float]] | None = None

    @model_validator(mode="after")
    def shape(self):
        if (self.quaternion_xyzw is None) == (self.rotation_matrix is None):
            raise ValueError("Provide quaternion or rotation matrix, exclusively")
        if self.rotation_matrix is not None:
            from scipy.spatial.transform import Rotation

            R = np.asarray(self.rotation_matrix)
            if (
                R.shape != (3, 3)
                or not np.allclose(R.T @ R, np.eye(3), atol=1e-5)
                or not np.isclose(np.linalg.det(R), 1)
            ):
                raise ValueError("Invalid rotation matrix")
            self.quaternion_xyzw = Rotation.from_matrix(R).as_quat().tolist()
            self.rotation_matrix = None
        if len(self.translation) != 3 or len(self.quaternion_xyzw) != 4:
            raise ValueError("Invalid pose vector lengths")
        if not np.isfinite(self.translation + self.quaternion_xyzw).all():
            raise ValueError("Pose values must be finite")
        if abs(np.linalg.norm(self.quaternion_xyzw) - 1) > 0.001:
            raise ValueError("Quaternion must be normalized")
        return self


class Poses(Strict):
    pose_convention: Literal["T_world_camera"] = "T_world_camera"
    translation_unit: Literal["metres", "millimetres", "reconstruction_units"]
    source_frame: str = "camera"
    destination_frame: str = "world"
    source: str = "unspecified"
    poses: list[Pose]

    @model_validator(mode="after")
    def ordering(self):
        if any(
            a.timestamp_ns >= b.timestamp_ns for a, b in zip(self.poses, self.poses[1:])
        ):
            raise ValueError("Pose timestamps must increase")
        return self


class DepthFrame(Strict):
    file: str
    timestamp_ns: int = Field(ge=0)
    rgb_frame_index: int = Field(ge=0)


class DepthMetadata(Strict):
    schema_version: Literal["1.0"] = "1.0"
    unit: Literal["metres", "millimetres", "relative"]
    width: int = Field(gt=0)
    height: int = Field(gt=0)
    invalid_value: float = 0
    camera: str
    rgb_aligned: bool
    T_rgb_depth: list[list[float]] | None = None
    scale_verified: bool = False
    frames: list[DepthFrame]

    @model_validator(mode="after")
    def alignment(self):
        if not self.rgb_aligned and self.T_rgb_depth is None:
            raise ValueError("Unaligned depth needs T_rgb_depth")
        if self.T_rgb_depth is not None:
            a = np.asarray(self.T_rgb_depth)
            if (
                a.shape != (4, 4)
                or not np.isfinite(a).all()
                or not np.allclose(a[3], [0, 0, 0, 1])
            ):
                raise ValueError("Invalid depth transform")
            rotation = a[:3, :3]
            if not np.allclose(
                rotation.T @ rotation, np.eye(3), atol=1e-5
            ) or not np.isclose(np.linalg.det(rotation), 1, atol=1e-5):
                raise ValueError(
                    "Depth transform rotation must be a proper rigid rotation"
                )
        return self


def resolve(path, root):
    root = Path(root).resolve()
    p = Path(path)
    resolved = (p if p.is_absolute() else root / p).resolve()
    if not resolved.is_relative_to(root):
        raise ValueError(f"Input path escapes project root: {path}")
    return resolved


def read_depth(path, metadata):
    path = Path(path)
    if path.suffix.lower() in [".npy", ".npz"]:
        arr = np.load(path, allow_pickle=False)
        if isinstance(arr, np.lib.npyio.NpzFile):
            try:
                if "depth" not in arr:
                    raise ValueError("NPZ must contain a depth array")
                depth = arr["depth"]
            finally:
                arr.close()
        else:
            depth = arr
    elif path.suffix.lower() in [".png", ".tif", ".tiff"]:
        depth = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        if depth is None:
            raise ValueError("Cannot decode depth image")
    else:
        raise ValueError("Unsupported depth format")
    if depth.shape != (metadata.height, metadata.width):
        raise ValueError("Depth dimensions disagree with metadata")
    depth = depth.astype(float)
    depth[depth == metadata.invalid_value] = np.nan
    if metadata.unit == "millimetres":
        depth /= 1000
    return depth


def read_imu(path):
    metadata_path = Path(path).with_suffix(".metadata.json")
    if not metadata_path.exists():
        raise ValueError(
            "IMU requires .metadata.json with units, time_offset_ns, and calibration"
        )
    metadata = json.loads(metadata_path.read_text())
    if (
        metadata.get("acceleration_unit") != "m/s^2"
        or metadata.get("angular_velocity_unit") != "rad/s"
    ):
        raise ValueError("Unsupported or missing IMU units")
    if "time_offset_ns" not in metadata or "calibration" not in metadata:
        raise ValueError("Missing IMU synchronization/calibration")
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        keys = ["timestamp_ns", "ax", "ay", "az", "gx", "gy", "gz"]
        if reader.fieldnames != keys:
            raise ValueError("Invalid IMU columns")
        rows = [[float(r[k]) for k in keys] for r in reader]
    arr = np.asarray(rows, dtype=float)
    if arr.ndim != 2 or not np.isfinite(arr).all() or np.any(np.diff(arr[:, 0]) <= 0):
        raise ValueError("Invalid IMU values/timestamps")
    return {"measurements": arr, "metadata": metadata, "fused": False}


def load_manifest(path, root):
    manifest = Manifest.model_validate_json(Path(path).read_text())
    resolved = {}
    warnings = []
    for name in ["before", "after"]:
        recording = getattr(manifest, name)
        files = {}
        for field, value in recording.model_dump().items():
            if value is None:
                files[field] = None
                continue
            p = resolve(value, root)
            if not p.exists():
                if field == "video":
                    raise FileNotFoundError(p)
                warnings.append(f"{name}.{field} absent; RGB-only fallback")
                files[field] = None
                continue
            files[field] = p
        if files["video"].suffix.lower() not in [".mp4", ".mov", ".mkv"]:
            raise ValueError("Supported video containers: MP4, MOV, MKV")
        if files["camera"]:
            files["camera_data"] = Camera.model_validate_json(
                files["camera"].read_text()
            )
        if files["poses"]:
            files["pose_data"] = read_poses(files["poses"])
        if files["imu"]:
            files["imu_data"] = read_imu(files["imu"])
        if files["depth_dir"]:
            directory = files["depth_dir"]
            meta = DepthMetadata.model_validate_json(
                (directory / "metadata.json").read_text()
            )
            Camera.model_validate_json(resolve(meta.camera, root).read_text())
            for frame in meta.frames:
                read_depth(resolve(str(directory / frame.file), root), meta)
            files["depth_metadata"] = meta
        if files["timestamps"]:
            times = json.loads(files["timestamps"].read_text())
            if not times or not np.isfinite(times).all() or np.any(np.diff(times) <= 0):
                raise ValueError("Invalid timestamp sidecar")
            files["timestamp_data"] = times
        resolved[name] = files
    annotation = None
    if manifest.annotations:
        ap = resolve(manifest.annotations, root)
        if ap.exists():
            annotation = Annotations.model_validate_json(ap.read_text())
        else:
            warnings.append(
                "Optional annotations unavailable; inference remains supported"
            )
        if annotation and annotation.scene_id != manifest.scene_id:
            raise ValueError("Annotation scene_id mismatch")
    return manifest, resolved, warnings


def read_poses(path):
    path = Path(path)
    if path.suffix.lower() == ".json":
        return Poses.model_validate_json(path.read_text())
    if path.suffix.lower() not in [".txt", ".npy"]:
        raise ValueError("Pose formats: JSON, TXT, NPY")
    metadata = json.loads(path.with_suffix(".metadata.json").read_text())
    arr = (
        np.load(path, allow_pickle=False)
        if path.suffix.lower() == ".npy"
        else np.loadtxt(path, ndmin=2)
    )
    if arr.ndim != 2 or arr.shape[1] != 8:
        raise ValueError("Pose array columns: timestamp_ns tx ty tz qx qy qz qw")
    poses = [
        Pose(
            timestamp_ns=int(row[0]),
            translation=row[1:4].tolist(),
            quaternion_xyzw=row[4:8].tolist(),
        )
        for row in arr
    ]
    return Poses(**metadata, poses=poses)


def validate_sensor_timing(files, video_manifest, tolerance_ns=50_000_000):
    times = np.asarray(video_manifest["original_timestamps_seconds"]) * 1e9
    depth = files.get("depth_metadata")
    if depth:
        for frame in depth.frames:
            if frame.rgb_frame_index >= len(times):
                raise ValueError("Depth references nonexistent RGB frame")
            if abs(frame.timestamp_ns - times[frame.rgb_frame_index]) > tolerance_ns:
                raise ValueError("Depth/RGB timestamp mismatch exceeds 50 ms")
    poses = files.get("pose_data")
    if poses and poses.poses:
        pts = np.array([p.timestamp_ns for p in poses.poses])
        if pts.max() < times.min() or pts.min() > times.max():
            raise ValueError("Pose and RGB timelines do not overlap")
    imu = files.get("imu_data")
    if imu:
        it = imu["measurements"][:, 0] + imu["metadata"]["time_offset_ns"]
        if it.max() < times.min() or it.min() > times.max():
            raise ValueError("Synchronized IMU and RGB timelines do not overlap")
