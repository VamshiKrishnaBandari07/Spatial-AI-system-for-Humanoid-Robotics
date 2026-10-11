from enum import Enum
from typing import Literal
from pydantic import BaseModel, Field, ConfigDict


class Change(str, Enum):
    ADDED = "ADDED"
    REMOVED = "REMOVED"
    MOVED = "MOVED"
    UNCHANGED = "UNCHANGED"
    UNKNOWN = "UNKNOWN"


class Config(BaseModel):
    model_config = ConfigDict(extra="forbid")
    max_frames: int = Field(80, ge=3, le=1000)
    max_width: int = Field(960, ge=64, le=4096)
    sample_seconds: float = Field(0.5, gt=0)
    focal_px: float | None = Field(None, gt=0)
    reprojection_px: float = Field(2.0, gt=0)
    registration_threshold: float = Field(0.08, gt=0)
    registration_min_heldout: int = Field(0, ge=0, le=1000)
    voxel_size: float = Field(0.08, gt=0)
    min_cluster: int = Field(5, ge=1)
    seed: int = 7
    geometry_backend: Literal["sparse", "vggt"] = "sparse"
    neural_frames_per_video: int = Field(2, ge=2, le=6)
    neural_frame_selection: Literal["temporal", "quality_diverse"] = "temporal"
    geometry_reuse_from: str | None = None
    metric_geometry_enabled: bool = False
    registration_initial: str | None = None
    evidence_calibration: str | None = None
    semantics_enabled: bool = False
    semantic_max_frames: int = Field(4, ge=2, le=24)
    max_detections_per_frame: int = Field(8, ge=1, le=50)
    detection_threshold: float = Field(0.35, ge=0, le=1)
    association_mode: Literal[
        "legacy", "appearance", "semantic", "semantic_geometry", "multiview"
    ] = "legacy"
    association_search_radius: float = Field(1.0, gt=0, allow_inf_nan=False)
    association_min_score: float = Field(0.8, gt=0, lt=1, allow_inf_nan=False)
    association_ambiguity_margin: float = Field(0.05, ge=0, le=1, allow_inf_nan=False)
    association_appearance_weight: float = Field(0.5, gt=0, allow_inf_nan=False)
    association_geometry_weight: float = Field(0.25, ge=0, allow_inf_nan=False)
    association_shape_weight: float = Field(0.15, ge=0, allow_inf_nan=False)
    association_view_weight: float = Field(0.1, ge=0, allow_inf_nan=False)
    movement_mode: Literal["legacy", "bounded"] = "legacy"
    movement_depth_error_bound: float | None = Field(None, ge=0, allow_inf_nan=False)
    movement_registration_error_bound: float | None = Field(
        None, ge=0, allow_inf_nan=False
    )
    movement_error_source: str | None = Field(None, min_length=1)
    identity_cosine: float = Field(0.85, ge=0, le=1)
    viewpoint_degrees: float = Field(10, gt=0, le=180)
    min_object_views: int = Field(2, ge=1, le=10)
    object_prompt: str = (
        "chair. table. box. backpack. bottle. plant. cabinet. electronic device."
    )


class Frame(BaseModel):
    id: str
    index: int
    timestamp: float
    path: str
    sharpness: float
    exposure: float
    features: int
    duplicate_distance: float | None = None


class Registration(BaseModel):
    valid: bool
    scale: float
    rotation: list[list[float]]
    translation: list[float]
    inliers: int
    correspondences: int
    rmse: float | None
    threshold: float
    reason: str
    heldout_count: int = 0
    heldout_inliers: int = 0
    heldout_rmse: float | None = None
    foreground_matches_excluded: int = 0
    metric_diagnostics: dict | None = None


class Event(BaseModel):
    schema_version: str = "1.0"
    id: str
    object_id: str | None = None
    semantic_label: str = "unknown region"
    change: Change
    before_position: list[float] | None = None
    after_position: list[float] | None = None
    coordinate_system: str = "before reconstruction; right-handed camera coordinates"
    scale_status: str = "unverified monocular scale"
    displacement: float | None = None
    displacement_units: str = "reconstruction units, not meters"
    supporting_frames: list[str] = Field(default_factory=list)
    evidence: dict = Field(default_factory=dict)
    contradictions: list[str] = Field(default_factory=list)
    decision_status: str = "candidate"
    explanation: str
