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


class SemanticExperts:
    def __init__(self):
        self.detector = Expert(
            ModelSpec(
                "IDEA-Research/grounding-dino-tiny",
                "a2bb814dd30d776dcf7e30523b00659f4f141c71",
                "detection",
            ),
            "/workspace/spatialmemory-models",
        ).load()
        self.appearance = Expert(
            ModelSpec(
                "facebook/dinov2-small",
                "ed25f3a31f01632728cabb09d1542f84ab7b0056",
                "appearance",
            ),
            "/workspace/spatialmemory-models",
        ).load()
        self.segmenter = Segmenter(SAM_REVISION)


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
        image = reconstruction.features[fid][2]
        h, w = image.shape[:2]
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        detections = experts.detector.infer(Image.fromarray(rgb), config.object_prompt)
        record = {"frame_id": fid, "state": state, "detections": detections}
        raw.append(record)
        boxes = detections["boxes"]
        scores = detections["raw_scores"]
        labels = detections["labels"]
        if not len(boxes) == len(scores) == len(labels):
            raise ValueError("Detector output lengths disagree")
        current_regions = []
        order = sorted(range(len(boxes)), key=lambda i: scores[i], reverse=True)[
            : config.max_detections_per_frame
        ]
        experts.segmenter.set_image(rgb)
        foreground[fid] = np.zeros((h, w), bool)
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
            foreground[fid] |= mask
            crop = rgb[y0:y1, x0:x1].copy()
            crop[~mask[y0:y1, x0:x1]] = 0
            appearance = experts.appearance.infer(Image.fromarray(crop))
            surface = reconstruction.surfaces.get(fid)
            if surface is not None:
                conf = surface["confidence"]
                valid = mask & (conf >= np.quantile(conf, 0.5))
                points = surface["points"][valid]
            else:
                ids = []
                for (frame, key), pid in reconstruction.tracks.items():
                    if frame != fid:
                        continue
                    x, y = np.rint(reconstruction.features[fid][0][key]).astype(int)
                    if 0 <= y < h and 0 <= x < w and mask[y, x]:
                        ids.append(pid)
                points = reconstruction.points[ids]
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
                str(labels[index]),
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
                "tracking": tracking,
                "state": state,
                "box": [int(v) for v in [x0, y0, x1, y1]],
                "detector": detections["provenance"],
                "raw_detector_score": float(scores[index]),
                "segmentation": segmentation,
                "appearance": appearance["provenance"],
                "location_point_count": len(points),
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
                    label=str(labels[index]),
                    position=center,
                    descriptor=appearance["descriptor"],
                    viewpoint=fid,
                    provenance=provenance,
                )
            )
        previous_image = image
        previous_regions = current_regions
    (output / f"{state}_model_outputs.json").write_text(json.dumps(raw, indent=2))
    return observations, foreground


def group_objects(observations, state, config):
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
        # Raw confidence only filters unreliable predictions; it is not a likelihood.
        depth[surface["confidence"] < np.quantile(surface["confidence"], 0.5)] = np.nan
        states = visibility(
            points,
            depth,
            K,
            pose,
            tolerance=config.voxel_size
            / max(registration.scale if registration else 1, 1e-9),
        )
        free = sum(x == "FREE" for x in states) / len(states)
        camera = -pose[:3, :3].T @ pose[:3, 3]
        ray = center - camera
        ray /= max(np.linalg.norm(ray), 1e-12)
        novel = all(
            np.degrees(np.arccos(np.clip(ray @ r, -1, 1))) >= config.viewpoint_degrees
            for r in rays
        )
        supported = free >= 0.9 and novel
        if supported:
            rays.append(ray)
        views.append(
            {
                "frame_id": fid,
                "states": states,
                "free_fraction": free,
                "counts_as_distinct_free_view": supported,
            }
        )
    return {
        "supported_free_views": len(rays),
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
        explanation = f"Model-proposed {source.label}: {classification.value.lower()}. "
        if classification == Change.UNKNOWN:
            explanation += "Distinct views, object identity, localization or positive free-space evidence are insufficient."
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
                and vis["supported_free_views"] >= config.min_object_views
            )
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
                {"distinct_views": obj.distinct_views, "visibility": vis},
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
