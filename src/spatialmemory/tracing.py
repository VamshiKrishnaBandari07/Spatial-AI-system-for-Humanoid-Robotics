"""Evidence diagnostics, never physical change predictions or calibrated scores."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal, Sequence
import json
import cv2
import numpy as np
from numpy.typing import NDArray

Status = Literal["PASS", "WARN", "FAIL", "NOT_RUN"]


@dataclass
class DiagnosticStepResult:
    object_id: str
    frame_idx: int
    stage_name: str
    status: Status
    metrics: dict[str, float] = field(default_factory=dict)
    notes: str = ""
    visual_artifact: NDArray[np.uint8] | None = field(default=None, repr=False)
    artifact_path: str | None = None


class PipelineTracer:
    """Collect diagnostics without changing detection, geometry or consensus gates.

    target_names are descriptive metadata, not detection prompts or answer keys.
    PASS means this check passed; it does not verify physical object state.
    """

    def __init__(self, target_names: Sequence[str] = ()) -> None:
        self.target_names = list(target_names)
        self.audit_log: list[DiagnosticStepResult] = []

    def _record(self, obj: str, frame: int, stage: str, status: Status,
                metrics: dict[str, float], notes: str,
                artifact: NDArray[np.uint8] | None = None) -> DiagnosticStepResult:
        row = DiagnosticStepResult(obj, frame, stage, status, metrics, notes, artifact)
        self.audit_log.append(row)
        return row

    def audit_crop_and_mask(self, obj_id: str, frame_idx: int, bbox: Sequence[float],
                            mask: NDArray, label_confidence: float,
                            image: NDArray[np.uint8] | None = None) -> DiagnosticStepResult:
        """bbox is native xyxy; mask is full-frame HxW, image optional HxWx3 BGR."""
        box = np.asarray(bbox, float)
        if (mask.ndim != 2 or mask.size == 0 or not np.isfinite(mask).all()
                or box.shape != (4,) or not np.isfinite(box).all()
                or not np.isfinite(label_confidence) or not 0 <= label_confidence <= 1
                or np.any(box[2:] <= box[:2])):
            return self._record(obj_id, frame_idx, "1_Crop_Mask", "FAIL", {}, "Invalid crop/mask/score evidence")
        h, w = mask.shape
        lo = np.maximum(np.floor(box[:2]).astype(int), 0)
        hi = np.minimum(np.ceil(box[2:]).astype(int), [w, h])
        if np.any(hi <= lo):
            return self._record(obj_id, frame_idx, "1_Crop_Mask", "FAIL", {}, "Box outside image")
        x0, y0 = lo
        x1, y1 = hi
        binary = mask > 0
        inside = int(binary[y0:y1, x0:x1].sum())
        total = int(binary.sum())
        coverage = inside / ((x1-x0)*(y1-y0))
        leakage = (total-inside) / max(1, total)
        status: Status = "PASS"
        notes = "Mask extent check passed; segmentation accuracy is unverified"
        if coverage < .2 or leakage > .1:
            status, notes = "WARN", "Sparse mask or outside-box leakage; inspect crop"
        if not total or not inside:
            status, notes = "FAIL", "No foreground inside box"
        elif label_confidence < .5:
            status = "WARN"
            notes += "; low raw detection score (uncalibrated)"
        artifact = None
        if image is not None:
            if image.shape != (h, w, 3) or image.dtype != np.uint8:
                raise ValueError("Diagnostic image must be native HxWx3 uint8 BGR")
            artifact = image[y0:y1, x0:x1].copy()
            selected = binary[y0:y1, x0:x1]
            artifact[selected] = (artifact[selected].astype(float)*.5 + np.array([0,255,0])*.5).astype(np.uint8)
        return self._record(obj_id, frame_idx, "1_Crop_Mask", status,
                            {"raw_detection_score": float(label_confidence),
                             "mask_coverage": float(coverage), "mask_leakage": float(leakage)}, notes, artifact)

    def audit_features_and_matching(self, obj_id: str, frame_idx: int,
                                    reid_embedding: NDArray, nearest_lookalike_dist: float | None,
                                    matched_distance: float | None = None) -> DiagnosticStepResult:
        """Distances are cosine distances [0,2] over nonzero normalized vectors."""
        embedding = np.asarray(reid_embedding, float)
        distances = [d for d in (nearest_lookalike_dist, matched_distance) if d is not None]
        if (embedding.ndim != 1 or not embedding.size or not np.isfinite(embedding).all()
                or np.linalg.norm(embedding) < 1e-12
                or any(not np.isfinite(d) or not 0 <= d <= 2 for d in distances)):
            return self._record(obj_id, frame_idx, "2_Tracking_Identity", "FAIL", {}, "Invalid embedding or cosine distance")
        metrics = {"embedding_norm": float(np.linalg.norm(embedding))}
        if nearest_lookalike_dist is not None:
            metrics["lookalike_embedding_dist"] = float(nearest_lookalike_dist)
        if matched_distance is not None:
            metrics["matched_embedding_dist"] = float(matched_distance)
        if nearest_lookalike_dist is None or matched_distance is None:
            return self._record(obj_id, frame_idx, "2_Tracking_Identity", "WARN", metrics, "Match/competitor evidence incomplete; identity not established")
        margin = nearest_lookalike_dist - matched_distance
        metrics["competitor_margin"] = float(margin)
        status: Status = "FAIL" if nearest_lookalike_dist < .15 or margin < .05 else "PASS"
        return self._record(obj_id, frame_idx, "2_Tracking_Identity", status, metrics, "Appearance diagnostic only; geometry and temporal identity gates still required")

    def audit_3d_localization(self, obj_id: str, frame_idx: int,
                              reprojection_error_px: float | Sequence[float],
                              threshold_px: float = 2.) -> DiagnosticStepResult:
        errors = np.asarray(reprojection_error_px, float).reshape(-1)
        if (not len(errors) or not np.isfinite(errors).all() or np.any(errors < 0)
                or not np.isfinite(threshold_px) or threshold_px <= 0):
            return self._record(obj_id, frame_idx, "3_3D_Processing", "FAIL", {}, "Invalid reprojection evidence")
        worst = float(errors.max())
        return self._record(obj_id, frame_idx, "3_3D_Processing", "PASS" if worst < threshold_px else "FAIL",
                            {"maximum_reprojection_px": worst, "threshold_px": float(threshold_px)},
                            "Pixel residual check only; cheirality, parallax and camera validity required separately")

    def audit_visibility_and_removal(self, obj_id: str, frame_idx: int,
                                     unobstructed_views_count: int, surface_empty_confidence: float,
                                     *, verified_view_ids: Sequence[str] = (),
                                     patches_verified: bool = False) -> DiagnosticStepResult:
        """Caller must verify distinct views; score is support, not calibrated confidence."""
        if (isinstance(unobstructed_views_count, bool) or not isinstance(unobstructed_views_count, int)
                or unobstructed_views_count < 0 or not np.isfinite(surface_empty_confidence)
                or not 0 <= surface_empty_confidence <= 1):
            return self._record(obj_id, frame_idx, "4_Visibility_Verification", "FAIL", {}, "Invalid visibility evidence")
        sufficient = (unobstructed_views_count >= 2 and len(set(verified_view_ids)) >= 2
                      and patches_verified and surface_empty_confidence >= .9)
        return self._record(obj_id, frame_idx, "4_Visibility_Verification", "PASS" if sufficient else "FAIL",
                            {"unobstructed_views": float(unobstructed_views_count), "empty_support": float(surface_empty_confidence)},
                            "Visibility evidence supported; not a removal verdict" if sufficient else "Insufficient verified visibility/occupancy evidence")

    def summary(self) -> dict:
        counts = {s: sum(r.status == s for r in self.audit_log) for s in ("PASS", "WARN", "FAIL", "NOT_RUN")}
        return {"status": "NOT_RUN" if not self.audit_log or counts["NOT_RUN"] == len(self.audit_log) else "FAIL" if counts["FAIL"] else "WARN" if counts["WARN"] or counts["NOT_RUN"] else "PASS",
                "counts": counts, "physical_changes_verified": False}

    def summarize_failures(self) -> None:
        print(json.dumps(self.summary()))
        for row in self.audit_log:
            if row.status != "PASS":
                print(f"[{row.stage_name}] {row.object_id} frame {row.frame_idx}: {row.status} | {row.notes}")

    def save(self, directory: Path) -> Path:
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        records = []
        for index, row in enumerate(self.audit_log):
            if row.visual_artifact is not None:
                filename = f"crop_mask_{index:06d}.png"
                if not cv2.imwrite(str(directory / filename), row.visual_artifact):
                    raise OSError("Cannot save diagnostic image")
                row.artifact_path = filename
            record = {k: v for k, v in row.__dict__.items() if k != "visual_artifact"}
            records.append(record)
        path = directory / "trace.json"
        path.write_text(json.dumps({"schema_version": "1.0", "target_names": self.target_names,
                                    "summary": self.summary(), "steps": records}, indent=2, allow_nan=False))
        return path


def export_run_trace(output: Path, destination: Path | None = None) -> Path:
    """Audit saved computed evidence read-only; never rerun models or infer labels.

    Uses exact frame IDs to locate native images. Frame indices here enumerate
    saved observations, not raw video frame numbers; original IDs are retained
    in object diagnostic identifiers. Missing stages remain NOT_RUN/WARN.
    """
    import sqlite3
    from .memory import Observation

    output = Path(output).resolve()
    tracer = PipelineTracer()
    database = output / "spatial_memory.sqlite"
    if database.exists():
        with sqlite3.connect(database.as_uri() + "?mode=ro", uri=True) as connection:
            observations = [Observation.model_validate_json(row[0]) for row in
                            connection.execute("SELECT data FROM observations ORDER BY id")]
        frames: dict[str, str] = {}
        for state in ("before", "after"):
            manifest = output / state / "manifest.json"
            if manifest.exists():
                frames.update({r["id"]: r["path"] for r in json.loads(manifest.read_text())["frames"]})
        for index, observation in enumerate(observations):
            p = observation.provenance
            mask_path = p.get("mask_file")
            mask_file = output / "evidence/supporting_frames" / Path(mask_path).name if mask_path else None
            mask = cv2.imread(str(mask_file), cv2.IMREAD_GRAYSCALE) if mask_file is not None and mask_file.is_file() else None
            image = cv2.imread(frames[observation.frame_id]) if observation.frame_id in frames else None
            if mask is not None and p.get("box") is not None and p.get("raw_detector_score") is not None:
                tracer.audit_crop_and_mask(observation.id, index, p["box"], mask,
                                           p["raw_detector_score"], image if image is not None and image.shape[:2] == mask.shape else None)
            else:
                tracer._record(observation.id, index, "1_Crop_Mask", "NOT_RUN", {}, "Native crop/mask evidence unavailable")
            tracer._record(observation.id, index, "2_Tracking_Identity", "WARN", {},
                           "See association.json for joint candidate costs and identity gates; no new identity inferred")
            rows = p.get("multiview_geometry", {}).get("candidate_pairs", [])
            errors = [r["maximum_accepted_reprojection_px"] for r in rows
                      if r.get("accepted") and r.get("maximum_accepted_reprojection_px") is not None]
            if errors:
                tracer.audit_3d_localization(observation.id, index, errors)
            else:
                tracer._record(observation.id, index, "3_3D_Processing", "NOT_RUN", {},
                               "No accepted per-point residual evidence; position alone cannot validate reprojection")
    association = output / "diagnostics/association.json"
    if association.exists():
        audit = json.loads(association.read_text())
        for state, objects in audit.get("visibility", {}).items():
            for object_id, visibility in objects.items():
                views = [v for v in visibility.get("views", []) if v.get("counts_as_distinct_free_view")]
                support = min((v.get("free_fraction", 0.) for v in views), default=0.)
                patches = bool(views) and all(
                    (v.get("crossview_surface_validation") or {}).get("supported_fraction", 0.) >= .9 for v in views)
                tracer.audit_visibility_and_removal(f"{state}:{object_id}", -1,
                    visibility.get("supported_free_views", 0), support,
                    verified_view_ids=[v["frame_id"] for v in views], patches_verified=patches)
    return tracer.save(destination or output / "diagnostics/tracer")
