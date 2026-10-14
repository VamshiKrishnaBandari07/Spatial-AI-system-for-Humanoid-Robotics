"""Explicit verified reuse of real neural observations; geometry is recomputed."""

from pathlib import Path
from typing import Sequence
import json
import shutil
import sqlite3
import cv2
import numpy as np

from .contracts import Config, Frame
from .geometry import Reconstruction
from .memory import Observation
from .ingestion import sha256
from .pixel_mapping import mask_to_geometry
from .association import unit_descriptor
from .surface_evidence import sample_surface


def cached_observations(
    source: Path,
    state: str,
    reconstruction: Reconstruction,
    frames: Sequence[Frame],
    output: Path,
    config: Config,
    video_sha256: str,
) -> tuple[list[Observation], dict[str, np.ndarray]]:
    """Reuse only matching videos, sampled pixels, masks and model configurations.

    Cached positions, camera centers, object IDs and geometry/identity beliefs
    are discarded. Fresh geometry and consensus cannot inherit old predictions.
    No neural call is claimed for stages reused from a completed source run.
    """
    from .semantics import localized_surface_points

    source = Path(source).resolve()
    output = Path(output)
    report = json.loads((source / "report.json").read_text())
    if report["status"] != "complete":
        raise ValueError("Perception cache requires a completed source run")
    index = 0 if state == "before" else 1
    if report["inputs"][index]["sha256"] != video_sha256:
        raise ValueError("Perception cache belongs to different videos or roles")
    keys = [
        "geometry_backend",
        "metric_geometry_enabled",
        "neural_preprocessing",
        "neural_frames_per_video",
        "neural_frame_selection",
        "semantic_max_frames",
        "max_frames",
        "max_width",
        "sample_seconds",
        "object_prompt",
        "detection_threshold",
        "max_detections_per_frame",
        "multiscale_detection_enabled",
        "detection_tile_size",
        "detection_max_tiles",
        "detection_max_refinements",
        "detector_temperature_file",
        "detector_variant",
        "semantic_label_validation_enabled",
        "semantic_label_margin",
    ]
    for key in keys:
        expected = report["config"].get(key, Config.model_fields[key].default)
        if expected != getattr(config, key):
            raise ValueError(f"Perception cache configuration differs: {key}")
    source_frames = {
        row["id"]: Frame.model_validate(row)
        for row in json.loads((source / state / "manifest.json").read_text())["frames"]
    }
    lookup = {frame.id: frame for frame in frames}
    with sqlite3.connect(
        (source / "spatial_memory.sqlite").as_uri() + "?mode=ro", uri=True
    ) as connection:
        saved = [
            Observation.model_validate_json(row[0])
            for row in connection.execute("SELECT data FROM observations")
        ]
    saved = [
        observation
        for observation in saved
        if observation.frame_id.startswith(state + ":")
    ]
    if not saved:
        raise ValueError("No cached neural observations for recording")
    evidence = output / "evidence/supporting_frames"
    evidence.mkdir(parents=True, exist_ok=True)
    foreground = {}
    observations = []
    verified_frames = set()
    mask_hashes = {}
    for observation in saved:
        fid = observation.frame_id
        if (
            fid not in reconstruction.poses
            or fid not in lookup
            or fid not in source_frames
        ):
            raise ValueError(
                "Cached observation frame is outside reconstruction coverage"
            )
        if fid not in verified_frames:
            if sha256(lookup[fid].path) != sha256(source_frames[fid].path):
                raise ValueError("Cached native frame pixels differ")
            verified_frames.add(fid)
        provenance = dict(observation.provenance)
        appearance = provenance.get("appearance", {})
        if (
            appearance.get("model") != "facebook/dinov2-small"
            or appearance.get("revision") != "ed25f3a31f01632728cabb09d1542f84ab7b0056"
        ):
            raise ValueError("Cached appearance model provenance differs")
        descriptor = unit_descriptor(observation.descriptor).tolist()
        mask_file = provenance["mask_file"]
        evidence_file = provenance["evidence_file"]
        if (
            Path(mask_file).name != mask_file
            or Path(evidence_file).name != evidence_file
        ):
            raise ValueError("Cached evidence path escapes permitted directory")
        mask = cv2.imread(
            str(source / "evidence/supporting_frames" / mask_file), cv2.IMREAD_GRAYSCALE
        )
        native = cv2.imread(lookup[fid].path)
        if mask is None or native is None or mask.shape != native.shape[:2]:
            raise ValueError("Cached mask coordinates disagree with native image")
        for name in [mask_file, evidence_file]:
            path = source / "evidence/supporting_frames" / name
            if not path.is_file():
                raise ValueError("Cached neural evidence file missing")
            shutil.copyfile(path, evidence / name)
        mask_hashes[mask_file] = sha256(evidence / mask_file)
        geometry_mask = mask_to_geometry(
            mask > 0,
            reconstruction.surfaces[fid]["depth"].shape,
            config.neural_preprocessing,
        )
        foreground.setdefault(fid, np.zeros(geometry_mask.shape, bool))
        foreground[fid] |= geometry_mask
        reconstruction.surfaces[fid]["max_reprojection_px"] = config.triangulation_max_reprojection_px
        points, localization = localized_surface_points(
            geometry_mask,
            reconstruction.surfaces[fid],
            reconstruction.poses[fid],
            config.voxel_size,
        )
        position = np.median(points, axis=0).tolist() if len(points) >= 8 else None
        extent = (
            np.quantile(points, [0.05, 0.95], axis=0).tolist()
            if position is not None
            else None
        )
        for key in ["identity_gallery", "multiview_geometry", "geometric_view_support"]:
            provenance.pop(key, None)
        provenance.update(
            observed_surface_samples=sample_surface(points),
            camera_center=(
                -reconstruction.poses[fid][:3, :3].T @ reconstruction.poses[fid][:3, 3]
            ).tolist(),
            localization=localization,
            location_point_count=len(points),
            extent=extent,
            perception_reuse={
                "source_report": str(source / "report.json"),
                "source_report_sha256": sha256(source / "report.json"),
                "origin": "Validated cached neural inference; not a fresh detector/SAM/ReID call",
            },
        )
        observations.append(
            observation.model_copy(
                update={
                    "object_id": None,
                    "position": position,
                    "descriptor": descriptor,
                    "provenance": provenance,
                }
            )
        )
    raw = json.loads((source / f"{state}_model_outputs.json").read_text())
    (output / f"{state}_model_outputs.json").write_text(json.dumps(raw, indent=2))
    directory = output / "diagnostics"
    directory.mkdir(exist_ok=True)
    (directory / f"{state}_perception_reuse.json").write_text(
        json.dumps(
            {
                "source_report": str(source / "report.json"),
                "source_report_sha256": sha256(source / "report.json"),
                "observations": len(observations),
                "verified_native_frames": len(verified_frames),
                "mask_sha256": mask_hashes,
                "fresh_neural_inference": False,
                "positions_recomputed": True,
            },
            indent=2,
        )
    )
    if config.multiview_geometry_enabled:
        from .object_geometry import recover_object_geometry

        observations = recover_object_geometry(
            observations, reconstruction, frames, output, config
        )
    return observations, foreground
