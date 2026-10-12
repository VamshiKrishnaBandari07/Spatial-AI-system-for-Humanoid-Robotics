"""Auditable object observations and conservative multi-view change consensus."""

from pathlib import Path
from dataclasses import dataclass
import json
import cv2, numpy as np
from .models import Expert, ModelSpec
from .segmentation import Segmenter
from .memory import Observation, Memory, associate, scene_graph
from .contracts import Event, Change
from .registration import transform
from .visibility import visibility
from .surface_evidence import sample_surface

SAM_REVISION = "de431c4043854a71d8101e17995dfe596bf101a5"


@dataclass
class ObjectState:
    id: str
    label: str
    position: list[float] | None
    descriptor: list[float] | None
    observations: list[Observation]
    extent: list[list[float]] | None
    distinct_views: int = 0
    position_spread: float | None = None
    identity_confirmed: bool = True


class SemanticExperts:
    def __init__(self, config=None):
        variant = config.detector_variant if config is not None else "tiny"
        detector_models = {
            "tiny": ("IDEA-Research/grounding-dino-tiny", "a2bb814dd30d776dcf7e30523b00659f4f141c71"),
            "base": ("IDEA-Research/grounding-dino-base", "12bdfa3120f3e7ec7b434d90674b3396eccf88eb"),
        }
        identifier, revision = detector_models[variant]
        self.detector = Expert(
            ModelSpec(
                identifier,
                revision,
                "detection",
            ),
            "/workspace/spatialmemory-models",
        ).load()
        if config is not None and config.detector_temperature_file:
            from .detector_calibration import DetectorTemperature
            calibration = json.loads(Path(config.detector_temperature_file).read_text())
            self.detector.temperature = DetectorTemperature(**calibration)
            self.detector.temperature.probabilities(np.zeros(1), revision)
        self.appearance = Expert(
            ModelSpec(
                "facebook/dinov2-small",
                "ed25f3a31f01632728cabb09d1542f84ab7b0056",
                "appearance",
            ),
            "/workspace/spatialmemory-models",
        ).load()
        self.segmenter = Segmenter(SAM_REVISION)
        self.labeler = None
        if config is not None and config.semantic_label_validation_enabled:
            from .semantic_labels import SemanticLabeler

            self.labeler = SemanticLabeler("/workspace/spatialmemory-models")


def independent_views(observations, position, minimum_degrees):
    if position is None:
        return 0
    directions = []
    for o in observations:
        center = o.provenance.get("camera_center")
        if center is None:
            continue
        ray = np.asarray(position) - center
        length = np.linalg.norm(ray)
        if length < 1e-9:
            continue
        ray = ray / length
        if all(
            np.degrees(np.arccos(np.clip(ray @ other, -1, 1))) >= minimum_degrees
            for other in directions
        ):
            directions.append(ray)
    return len(directions)


def localized_surface_points(mask, surface, pose, tolerance, return_support=False):
    """Keep the point-head gate; admit depth-head support only with geometric agreement."""
    points = surface["points"]
    confidence = surface["confidence"]
    content = surface.get("valid_pixels", np.ones(mask.shape, bool))
    valid = mask & content & (confidence >= np.quantile(confidence[content], 0.5))
    primary_count = int(valid.sum())
    depth_confidence = surface.get("depth_confidence")
    alternate_count = 0
    if depth_confidence is not None:
        depth = surface["depth"]
        if depth_confidence.shape != mask.shape or depth.shape != mask.shape:
            raise ValueError("Depth-head dimensions disagree with semantic mask")
        y, x = np.indices(mask.shape)
        pixels = np.stack([x, y, np.ones_like(x)], axis=-1)
        camera_points = (
            pixels @ np.linalg.inv(surface["intrinsics"]).T * depth[..., None]
        )
        # Poses are T_camera_world; convert camera depth points into this session's world.
        depth_points = (camera_points - pose[:3, 3]) @ pose[:3, :3]
        agreement = np.linalg.norm(depth_points - points, axis=-1)
        supported = (
            mask
            & content
            & np.isfinite(depth)
            & (depth > 0)
            & np.isfinite(depth_points).all(axis=-1)
            & (depth_confidence >= np.quantile(depth_confidence[content], 0.5))
            & (agreement <= tolerance)
        )
        alternate_count = int((supported & ~valid).sum())
        valid |= supported
    selected = points[valid]
    selected = selected[np.isfinite(selected).all(axis=1)]
    audit = {
        "point_head_supported_pixels": primary_count,
        "additional_depth_head_agreement_pixels": alternate_count,
        "depth_head_available": depth_confidence is not None,
        "agreement_tolerance_reconstruction_units": tolerance,
        "confidence_gate": "per-head frame median; raw scores are not calibrated",
        "independent_geometry_validation": False,
    }
    return (selected, audit, valid) if return_support else (selected, audit)


def observe_scene(reconstruction, frames, state, output, config, experts):
    """Store model outputs, masks, locations and model-proposed labels separately."""
    from PIL import Image

    output = Path(output)
    evidence_dir = output / "evidence/supporting_frames"
    evidence_dir.mkdir(parents=True, exist_ok=True)
    frame_lookup = {f.id: f for f in frames}
    available = list(reconstruction.poses)
    chosen = [
        available[i]
        for i in np.linspace(
            0, len(available) - 1, min(len(available), config.semantic_max_frames)
        )
        .round()
        .astype(int)
    ]
    if config.neural_frame_selection == "quality_diverse":
        from .ingestion import neural_keyframes

        chosen = [
            f.id
            for f in neural_keyframes(
                [frame_lookup[fid] for fid in available], config.semantic_max_frames
            )
        ]
    observations = []
    foreground = {}
    raw = []
    previous_image = None
    previous_regions = []
    for fid in chosen:
        geometry_image = reconstruction.features[fid][2]
        image = geometry_image
        if config.geometry_backend == "vggt" and not config.metric_geometry_enabled:
            image = cv2.imread(frame_lookup[fid].path)
            if image is None:
                raise ValueError("Cannot read original perception frame")
        h, w = image.shape[:2]
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        if config.multiscale_detection_enabled:
            from .multiscale_detection import multiscale_detect

            detections = multiscale_detect(
                experts.detector, Image.fromarray(rgb), config.object_prompt,
                config.detection_tile_size, config.detection_max_tiles,
                config.detection_max_refinements,
            )
        else:
            detections = experts.detector.infer(Image.fromarray(rgb), config.object_prompt)
        record = {"frame_id": fid, "state": state, "detections": detections}
        record["perception_size"] = [w, h]
        record["geometry_size"] = list(geometry_image.shape[:2][::-1])
        raw.append(record)
        boxes = detections["boxes"]
        scores = detections["raw_scores"]
        labels = detections["labels"]
        if not len(boxes) == len(scores) == len(labels):
            raise ValueError("Detector output lengths disagree")
        current_regions = []
        from .detection_selection import select_detections

        order = select_detections(
            boxes,
            scores,
            labels,
            config.detection_threshold,
            config.max_detections_per_frame,
        )
        record["selected_detection_indices"] = order
        record["selection_method"] = (
            "score-ranked, same-category duplicate suppression at IoU 0.8"
        )
        experts.segmenter.set_image(rgb)
        foreground[fid] = np.zeros(geometry_image.shape[:2], bool)
        for index in order:
            if scores[index] < config.detection_threshold:
                continue
            box = np.asarray(boxes[index], float)
            if box.shape != (4,) or not np.isfinite(box).all():
                raise ValueError("Invalid detection box")
            x0, y0, x1, y1 = np.rint(
                np.clip(box, [0, 0, 0, 0], [w - 1, h - 1, w, h])
            ).astype(int)
            if x1 <= x0 or y1 <= y0:
                continue
            mask, segmentation = experts.segmenter.mask([x0, y0, x1, y1])
            if mask.shape != (h, w):
                raise ValueError("Segmentation dimensions disagree with image")
            if not mask.any():
                continue
            # Keep prompt leakage outside the box from dominating geometry/registration.
            inside = np.zeros_like(mask)
            inside[y0:y1, x0:x1] = True
            mask = mask & inside
            if mask.sum() < 16:
                continue
            geometry_mask = mask
            if config.geometry_backend == "vggt" and not config.metric_geometry_enabled:
                from .pixel_mapping import mask_to_geometry

                geometry_mask = mask_to_geometry(
                    mask, geometry_image.shape[:2], config.neural_preprocessing
                )
            foreground[fid] |= geometry_mask
            from .appearance import masked_crop

            crop = masked_crop(rgb, mask, [x0, y0, x1, y1])
            appearance = experts.appearance.infer(Image.fromarray(crop))
            effective_label = str(labels[index])
            semantic_check = None
            if config.semantic_label_validation_enabled:
                semantic_check = experts.labeler.infer(
                    Image.fromarray(rgb[y0:y1, x0:x1]), config.semantic_label_margin,
                )
                effective_label = semantic_check["label"]
            surface = reconstruction.surfaces.get(fid)
            if surface is not None:
                points, localization = localized_surface_points(
                    geometry_mask, surface, reconstruction.poses[fid], config.voxel_size
                )
            else:
                ids = []
                for (frame, key), pid in reconstruction.tracks.items():
                    if frame != fid:
                        continue
                    x, y = np.rint(reconstruction.features[fid][0][key]).astype(int)
                    if 0 <= y < h and 0 <= x < w and mask[y, x]:
                        ids.append(pid)
                points = reconstruction.points[ids]
                localization = {"source": "sparse tracks; depth head unavailable"}
            points = points[np.isfinite(points).all(axis=1)]
            center = np.median(points, axis=0).tolist() if len(points) >= 8 else None
            extent = (
                np.quantile(points, [0.05, 0.95], axis=0).tolist()
                if center is not None
                else None
            )
            camera = (
                -reconstruction.poses[fid][:3, :3].T @ reconstruction.poses[fid][:3, 3]
            )
            oid = f"{state}_{fid.split(':')[-1]}_{index}"
            annotated = image.copy()
            annotated[mask] = (
                annotated[mask] * 0.45 + np.array([40, 180, 250]) * 0.55
            ).astype(np.uint8)
            cv2.rectangle(annotated, (x0, y0), (x1, y1), (255, 180, 30), 2)
            cv2.putText(
                annotated,
                effective_label,
                (x0, max(15, y0 - 6)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                1,
            )
            evidence_file = f"{oid}.png"
            mask_file = f"{oid}_mask.png"
            cv2.imwrite(str(evidence_dir / evidence_file), annotated)
            cv2.imwrite(str(evidence_dir / mask_file), mask.astype(np.uint8) * 255)
            tracking = {
                "status": "initial observation",
                "physical_motion_verified": False,
            }
            previous = [r for r in previous_regions if r["label"] == str(labels[index])]
            if (
                previous_image is not None
                and previous_image.shape == image.shape
                and len(previous) == 1
                and list(map(str, labels)).count(str(labels[index])) == 1
            ):
                from .segmentation import propagate_mask

                propagated, flow = propagate_mask(
                    previous_image, image, previous[0]["mask"]
                )
                overlap = float(
                    (propagated & mask).sum() / max(1, (propagated | mask).sum())
                )
                tracking = {
                    "status": "consistent"
                    if overlap >= 0.3 and not flow["requires_reinitialization"]
                    else "reinitialized from detector/SAM 2",
                    "mask_iou": overlap,
                    "flow": flow,
                    "physical_motion_verified": False,
                }
            current_regions.append({"label": str(labels[index]), "mask": mask})
            provenance = {
                "raw_detector_label": str(labels[index]),
                "detector_probability": (detections.get("calibrated_scores") or [None] * len(boxes))[index],
                "detector_calibration_source": detections.get("calibration_source"),
                "semantic_cross_check": semantic_check,
                "tracking": tracking,
                "state": state,
                "box": [int(v) for v in [x0, y0, x1, y1]],
                "detector": detections["provenance"],
                "raw_detector_score": float(scores[index]),
                "segmentation": segmentation,
                "appearance": appearance["provenance"],
                "location_point_count": len(points),
                "observed_surface_samples": sample_surface(points),
                "localization": localization,
                "extent": extent,
                "camera_center": camera.tolist(),
                "image_size": [w, h],
                "evidence_file": evidence_file,
                "mask_file": mask_file,
                "semantic_status": "model-proposed label, not verified ground truth",
                "location_uncertainty": "unquantified reconstruction/segmentation error; extent is not covariance",
            }
            observations.append(
                Observation(
                    id=oid,
                    frame_id=fid,
                    timestamp=frame_lookup[fid].timestamp,
                    label=effective_label,
                    position=center,
                    descriptor=appearance["descriptor"],
                    viewpoint=fid,
                    provenance=provenance,
                )
            )
        previous_image = image
        previous_regions = current_regions
    (output / f"{state}_model_outputs.json").write_text(json.dumps(raw, indent=2))
    if config.multiview_geometry_enabled:
        from .object_geometry import recover_object_geometry

        observations = recover_object_geometry(
            observations, reconstruction, frames, output, config,
        )
    return observations, foreground


def group_objects(observations, state, config, identity_checkpoint=None):
    if config.association_mode == "joint":
        from .identity_integration import group_with_gallery

        return group_with_gallery(observations, state, config, identity_checkpoint)
    groups = []
    # Per-frame assignment prevents two identical detections in a frame from merging.
    for frame in dict.fromkeys(o.frame_id for o in observations):
        current = [o for o in observations if o.frame_id == frame]
        representatives = [
            Observation(
                id=g.id,
                frame_id="fused",
                timestamp=0,
                label=g.label,
                position=g.position,
                descriptor=g.descriptor,
                viewpoint="fused",
            )
            for g in groups
        ]
        matches = associate(
            representatives, current, max_cost=1 - config.identity_cosine
        )
        if config.association_mode != "legacy":
            from .association import match_objects

            matches = match_objects(groups, current, config, within_recording=True)
        used = set()
        for m in matches:
            g = groups[m["before"]]
            o = current[m["after"]]
            if not m["identity_verified"] or o.position is None or g.position is None:
                continue
            # Within one recording, geometry helps reject appearance-only identity merges.
            if (
                np.linalg.norm(np.asarray(o.position) - g.position)
                > config.voxel_size * 5
            ):
                continue
            g.observations.append(o)
            used.add(m["after"])
            positions = [x.position for x in g.observations if x.position is not None]
            g.position = np.median(positions, axis=0).tolist()
            vector = np.mean([x.descriptor for x in g.observations], axis=0)
            vector /= max(np.linalg.norm(vector), 1e-12)
            g.descriptor = vector.tolist()
        for i, o in enumerate(current):
            if i not in used:
                groups.append(
                    ObjectState(
                        f"{state}_object_{len(groups)}",
                        o.label,
                        o.position,
                        o.descriptor,
                        [o],
                        o.provenance.get("extent"),
                    )
                )
    for g in groups:
        g.distinct_views = independent_views(
            g.observations, g.position, config.viewpoint_degrees
        )
        if g.position is not None:
            positions = np.asarray(
                [o.position for o in g.observations if o.position is not None]
            )
            g.position_spread = float(
                np.max(np.linalg.norm(positions - g.position, axis=1))
            )
    return groups


def common_frame(observations, registration):
    R = np.array(registration.rotation)
    t = np.array(registration.translation)
    s = registration.scale
    result = []
    for o in observations:
        provenance = dict(o.provenance)
        if provenance.get("observed_surface_samples"):
            provenance["observed_surface_samples"] = transform(
                np.asarray(provenance["observed_surface_samples"]), s, R, t
            ).tolist()
        provenance["camera_center"] = transform(
            np.array([provenance["camera_center"]]), s, R, t
        )[0].tolist()
        if provenance.get("extent") is not None:
            lo, hi = np.array(provenance["extent"])
            corners = np.array(
                [
                    [x, y, z]
                    for x in [lo[0], hi[0]]
                    for y in [lo[1], hi[1]]
                    for z in [lo[2], hi[2]]
                ]
            )
            corners = transform(corners, s, R, t)
            provenance["extent"] = [corners.min(0).tolist(), corners.max(0).tolist()]
        position = (
            transform(np.array([o.position]), s, R, t)[0].tolist()
            if o.position is not None
            else None
        )
        result.append(
            o.model_copy(update={"position": position, "provenance": provenance})
        )
    return result


def visibility_evidence(obj, reconstruction, config, registration=None):
    if obj.position is None:
        return {"supported_free_views": 0, "views": [], "source": "no localized object"}
    samples = [obj.position]
    if obj.extent is not None:
        lo, hi = np.array(obj.extent)
        samples.extend(
            [
                [x, y, z]
                for x in [lo[0], hi[0]]
                for y in [lo[1], hi[1]]
                for z in [lo[2], hi[2]]
            ]
        )
    sample_source = "centroid and box corners"
    if config.observed_surface_visibility_enabled:
        from .surface_evidence import sample_surface
        observed = [point for observation in obj.observations
                    for point in observation.provenance.get("observed_surface_samples", [])]
        if len(observed) < 4:
            return {"supported_free_views": 0, "views": [], "source": "insufficient observed surface samples"}
        samples = sample_surface(np.asarray(observed))
        sample_source = "actual computed object-surface points; geometry uncertainty uncalibrated"
    points = np.asarray(samples)
    center = np.asarray(obj.position)
    views = []
    rays = []
    if registration is not None:
        R = np.array(registration.rotation)
        t = np.array(registration.translation)
        s = registration.scale
        points = (points - t) @ R / s
        center = (center - t) @ R / s
    for fid, surface in reconstruction.surfaces.items():
        pose = reconstruction.poses[fid]
        K = surface["intrinsics"]
        depth = surface["depth"].copy()
        # Use the same cross-head support policy for localization and visibility.
        # Cache per tolerance; the session's surfaces/poses are immutable here.
        tolerance = config.voxel_size / max(
            registration.scale if registration else 1, 1e-9
        )
        support_cache = surface.setdefault("_visibility_support", {})
        if tolerance not in support_cache:
            if surface.get("depth_confidence") is None:
                # Legacy depth-only surfaces retain their original confidence gate.
                content = surface.get("valid_pixels", np.ones(depth.shape, bool))
                support = content & (
                    surface["confidence"]
                    >= np.quantile(surface["confidence"][content], 0.5)
                )
            else:
                _, _, support = localized_surface_points(
                    np.ones(depth.shape, bool),
                    surface,
                    pose,
                    tolerance,
                    return_support=True,
                )
            support_cache[tolerance] = support
        depth[~support_cache[tolerance]] = np.nan
        states = visibility(
            points,
            depth,
            K,
            pose,
            tolerance=tolerance,
        )
        from .visibility import visibility_details
        ray_diagnostics = visibility_details(points, surface["depth"], K, pose,
                                              support_cache[tolerance], tolerance)
        free = sum(x == "FREE" for x in states) / len(states)
        crossview_support = None
        minimum_angle = config.viewpoint_degrees
        if getattr(config, "geometry_supported_views_enabled", False) and free >= 0.9:
            from .visibility import crossview_free_surface

            crossview_support = crossview_free_surface(
                points, fid, reconstruction, tolerance, config.triangulation_min_degrees,
            )
            if crossview_support["supported_fraction"] >= 0.9:
                minimum_angle = config.triangulation_min_degrees
        camera = -pose[:3, :3].T @ pose[:3, 3]
        ray = center - camera
        ray /= max(np.linalg.norm(ray), 1e-12)
        novel = all(
            np.degrees(np.arccos(np.clip(ray @ r, -1, 1))) >= minimum_angle
            for r in rays
        )
        # A missing box is never occupancy evidence. For predicted geometry,
        # require explicit multi-view textured background confirmation too.
        patch_verified = (crossview_support is not None
                          and crossview_support["supported_fraction"] >= 0.9)
        supported = free >= 0.9 and novel and (
            not getattr(config, "geometry_supported_views_enabled", False)
            or patch_verified
        )
        if supported:
            rays.append(ray)
        views.append(
            {
                "frame_id": fid,
                "states": states,
                "ray_diagnostics": ray_diagnostics,
                "free_fraction": free,
                "counts_as_distinct_free_view": supported,
                "crossview_surface_validation": crossview_support,
                "minimum_view_angle_degrees": minimum_angle,
            }
        )
    return {
        "supported_free_views": len(rays),
        "sample_source": sample_source,
        "sample_count": len(points),
        "views": views,
        "source": "Measured calibrated registered depth"
        if getattr(reconstruction, "diagnostics", {}).get("sensor_geometry_used")
        else "VGGT predicted registered depth; accuracy/uncertainty not calibrated",
        "metric": bool(
            getattr(reconstruction, "diagnostics", {}).get("sensor_geometry_used")
        ),
    }


def consensus(before, after, b_reconstruction, a_reconstruction, registration, config):
    representatives = lambda groups: [
        Observation(
            id=g.id,
            frame_id="fused",
            timestamp=0,
            label=g.label,
            position=g.position,
            descriptor=g.descriptor,
            viewpoint="fused",
        )
        for g in groups
    ]
    associations = associate(
        representatives(before),
        representatives(after),
        max_cost=1 - config.identity_cosine,
    )
    if config.association_mode != "legacy":
        from .association import match_objects

        associations = match_objects(before, after, config)
    events = []
    used_before = set()
    used_after = set()
    calibration = None
    if config.evidence_calibration:
        from .calibration import LikelihoodCalibration

        labeled = json.loads(Path(config.evidence_calibration).read_text())
        calibration = LikelihoodCalibration.fit(labeled["records"], labeled["source"])

    def create(a, b, classification, status, evidence, contradictions=None):
        source = a or b
        frames = [o.frame_id for g in [a, b] if g for o in g.observations]
        observations = [o.model_dump() for g in [a, b] if g for o in g.observations]
        movement = (
            float(np.linalg.norm(np.asarray(a.position) - b.position))
            if a and b and a.position is not None and b.position is not None
            else None
        )
        evidence.update(
            observations=observations,
            before_extent=a.extent if a else None,
            after_extent=b.extent if b else None,
            registration=registration.model_dump(),
            posterior_probability=None,
            calibrated=False,
            consensus_method="auditable rule consensus; no calibrated likelihoods supplied",
        )
        if calibration is not None:
            from .evidence import Evidence, fuse

            # Missing signals contribute nothing; caller-supplied training data supplies likelihoods.
            signals = {
                "unique_identity": bool(
                    evidence.get("association", {}).get("identity_verified")
                ),
                "viewpoint_diversity": all(
                    g.distinct_views >= config.min_object_views for g in [a, b] if g
                ),
                "free_space": evidence.get("visibility", {}).get(
                    "supported_free_views", 0
                )
                >= config.min_object_views,
                "geometric_displacement": movement is not None
                and movement > config.voxel_size * 2,
            }
            terms = []
            for name, present in signals.items():
                if name in calibration.rates:
                    terms.append(
                        Evidence(
                            "object-consensus",
                            calibration.likelihood_ratio(name, present),
                            name,
                        )
                    )
            fused = fuse(terms)
            evidence["probabilistic_fusion"] = {
                "result": fused,
                "calibration": calibration.audit(),
                "hypothesis": "physical change present",
                "all_signals_share_a_correlation_group": True,
                "posterior_is_not_deployment_calibrated": True,
            }
        gates = []
        if classification == Change.UNKNOWN:
            if any(g.position is None for g in [a, b] if g):
                gates.append("missing_object_localization")
            if any(g.distinct_views < config.min_object_views for g in [a, b] if g):
                gates.append("insufficient_independent_views")
            if evidence.get("association") and not evidence["association"].get(
                "identity_verified"
            ):
                gates.append("ambiguous_identity_assignment")
            if evidence.get("unresolved_identity_candidates"):
                gates.append("unresolved_cross_recording_identity")
            if (
                "visibility" in evidence
                and evidence["visibility"].get("supported_free_views", 0)
                < config.min_object_views
            ):
                gates.append("insufficient_positive_free_space")
            movement_gate = evidence.get("bounded_movement", {}).get("blocking_gate")
            if movement_gate:
                gates.append(movement_gate)
            if not gates:
                gates.append("displacement_or_identity_evidence_inconclusive")
        evidence["decision_gates"] = gates
        evidence["validation"] = {
            "status": "failed" if gates else "rule_gates_passed",
            "failures": gates,
            "verified_physical_state": False,
            "calibrated_state_confidence": None,
        }
        explanation = f"Model-proposed {source.label}: {classification.value.lower()}. "
        if classification == Change.UNKNOWN:
            explanation += "Decision blocked by: " + ", ".join(gates) + "."
        elif classification == Change.MOVED:
            explanation += f"Unique appearance association and registered geometry suggest displacement of {movement:.3g} reconstruction units; absolute scale and model error are unverified."
        elif classification == Change.UNCHANGED:
            explanation += "Appearance association and registered centroids agree within the configured geometric threshold across distinct viewpoints."
        else:
            explanation += "Multiple distinct viewpoints support a predicted free-space transition; this remains probable because depth and semantic outputs are model predictions."
        events.append(
            Event(
                id=f"object-event-{len(events)}",
                object_id=source.id,
                semantic_label=source.label,
                change=classification,
                before_position=a.position if a else None,
                after_position=b.position if b else None,
                displacement=movement,
                supporting_frames=list(dict.fromkeys(frames)),
                evidence=evidence,
                contradictions=contradictions or [],
                decision_status=status,
                explanation=explanation,
            )
        )

    for match in associations:
        i, j = match["before"], match["after"]
        a = before[i]
        b = after[j]
        used_before.add(i)
        used_after.add(j)
        enough = (
            a.distinct_views >= config.min_object_views
            and b.distinct_views >= config.min_object_views
        )
        localized = a.position is not None and b.position is not None
        change = Change.UNKNOWN
        status = "ambiguous"
        details = {
            "association": match,
            "before_distinct_views": a.distinct_views,
            "after_distinct_views": b.distinct_views,
        }
        if match["identity_verified"] and enough and localized:
            displacement = np.linalg.norm(np.asarray(a.position) - b.position)
            threshold = config.voxel_size * 2
            band = max(
                registration.rmse or 0, a.position_spread or 0, b.position_spread or 0
            )
            details.update(
                displacement_threshold=threshold,
                observed_centroid_spread_band=band,
                spread_is_calibrated_uncertainty=False,
            )
            if displacement > threshold + band:
                change = Change.MOVED
                status = "probable"
            elif displacement < threshold - band:
                change = Change.UNCHANGED
                status = "supported"
            if config.movement_mode == "bounded":
                from .movement import bounded_movement

                result, movement_details = bounded_movement(a, b, config)
                change = Change(result)
                status = "ambiguous" if change == Change.UNKNOWN else "probable"
                details["bounded_movement"] = movement_details
            b.id = a.id
        contradictions = (
            []
            if match["identity_verified"]
            else ["Competing appearance identities retained; assignment is ambiguous"]
        )
        create(a, b, change, status, details, contradictions)
    for state, groups, used, other in [
        ("before", before, used_before, a_reconstruction),
        ("after", after, used_after, b_reconstruction),
    ]:
        for i, obj in enumerate(groups):
            if i in used:
                continue
            vis = visibility_evidence(
                obj, other, config, registration if state == "before" else None
            )
            supported = (
                obj.distinct_views >= config.min_object_views
                and vis["supported_free_views"] >= max(2, config.min_object_views)
            )
            unresolved = []
            if config.association_mode != "legacy":
                from .association import semantic_label

                other_groups = after if state == "before" else before
                other_used = used_after if state == "before" else used_before
                for j, candidate in enumerate(other_groups):
                    if j in other_used or semantic_label(obj.label) != semantic_label(
                        candidate.label
                    ):
                        continue
                    if (
                        obj.position is None
                        or candidate.position is None
                        or np.linalg.norm(np.asarray(obj.position) - candidate.position)
                        <= config.association_search_radius
                    ):
                        unresolved.append(candidate.id)
                if unresolved:
                    supported = False
            change = (
                (Change.REMOVED if state == "before" else Change.ADDED)
                if supported
                else Change.UNKNOWN
            )
            create(
                obj if state == "before" else None,
                obj if state == "after" else None,
                change,
                "probable" if supported else "unobservable",
                {
                    "distinct_views": obj.distinct_views,
                    "visibility": vis,
                    "unresolved_identity_candidates": unresolved,
                },
            )
    return events, associations


def persist_memory(output, before, after, events):
    output = Path(output)
    path = output / "spatial_memory.sqlite"
    if path.exists():
        raise FileExistsError(
            "Spatial memory already exists; use a fresh output directory"
        )
    memory = Memory(path)
    try:
        for state, groups in [("before", before), ("after", after)]:
            for g in groups:
                for o in g.observations:
                    memory.observe(o.model_copy(update={"object_id": g.id}))
            states = [
                {
                    "id": g.id,
                    "label": g.label,
                    "position": g.position,
                    "extent": g.extent,
                    "distinct_views": g.distinct_views,
                    "observation_ids": [o.id for o in g.observations],
                }
                for g in groups
            ]
            graph = scene_graph(
                [
                    Observation(
                        id=g.id,
                        object_id=g.id,
                        frame_id="fused",
                        timestamp=0,
                        label=g.label,
                        position=g.position,
                        viewpoint="fused",
                    )
                    for g in groups
                ]
            )
            memory.snapshot(
                state,
                {"objects": states, "graph": graph, "beliefs_are_ground_truth": False},
            )
        payload = {
            "before": memory.world("before"),
            "after": memory.world("after"),
            "events": [e.model_dump(mode="json") for e in events],
        }
        (output / "spatial_memory.json").write_text(json.dumps(payload, indent=2))
        return payload
    finally:
        memory.close()
