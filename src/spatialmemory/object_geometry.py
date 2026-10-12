"""Object geometry from native image correspondences and checked camera rays.

This cross-check is independent of point-head confidence ranking, but still
uses predicted poses/intrinsics; it is not metric calibration or ground truth.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import cv2
import numpy as np
from numpy.typing import NDArray

from .association import semantic_label, unit_descriptor
from .contracts import Config, Frame
from .geometry import Reconstruction, match, project
from .memory import Observation
from .pixel_mapping import native_to_geometry_matrix
from .reprojection import refine_point

Array = NDArray[np.float64]


@dataclass
class TriangulationResult:
    points: Array
    valid: NDArray[np.bool_]
    reprojection_errors: Array
    angles_degrees: Array


def triangulate_checked(
    pixels_a: Array,
    pixels_b: Array,
    ka: Array,
    kb: Array,
    pose_a: Array,
    pose_b: Array,
    max_error: float,
    min_angle: float,
) -> TriangulationResult:
    """Triangulate Nx2 matches in world coordinates; poses are T_camera_world.

    Require positive depth in both views, finite coordinates, bounded native
    pixel reprojection residual and angular parallax. Invalid points cannot
    contribute a centroid or extent. Co-located camera centers fail closed.
    """
    if pixels_a.shape != pixels_b.shape or pixels_a.ndim != 2 or pixels_a.shape[1] != 2:
        raise ValueError("Correspondences must have matching Nx2 dimensions")
    if max_error <= 0 or min_angle <= 0:
        raise ValueError("Positive triangulation error/angle gates required")
    for k, pose in [(ka, pose_a), (kb, pose_b)]:
        if (
            k.shape != (3, 3)
            or pose.shape != (4, 4)
            or not np.isfinite(k).all()
            or not np.isfinite(pose).all()
        ):
            raise ValueError("Invalid camera calibration or pose")
        if k[0, 0] <= 0 or k[1, 1] <= 0 or not np.allclose(k[2], [0, 0, 1], atol=1e-6):
            raise ValueError("Invalid pinhole intrinsics")
        if (
            not np.allclose(pose[3], [0, 0, 0, 1], atol=1e-6)
            or not np.allclose(pose[:3, :3] @ pose[:3, :3].T, np.eye(3), atol=1e-3)
            or np.linalg.det(pose[:3, :3]) <= 0
        ):
            raise ValueError(
                "Pose must be a right-handed rigid T_camera_world transform"
            )
    if not np.isfinite(pixels_a).all() or not np.isfinite(pixels_b).all():
        raise ValueError("Nonfinite pixel correspondences")
    count = len(pixels_a)
    if count == 0:
        return TriangulationResult(
            np.empty((0, 3)), np.zeros(0, bool), np.empty((0, 2)), np.empty(0)
        )
    homogeneous = cv2.triangulatePoints(
        ka @ pose_a[:3],
        kb @ pose_b[:3],
        pixels_a.T.astype(float),
        pixels_b.T.astype(float),
    ).T
    with np.errstate(divide="ignore", invalid="ignore"):
        points = homogeneous[:, :3] / homogeneous[:, 3, None]
        # Optimize before the final gate. Initial residuals only bound work;
        # they cannot authorize geometry. Gross mismatches remain rejected.
        for i, point in enumerate(points):
            if np.isfinite(point).all() and np.linalg.norm(point) < 1e9:
                fitted = refine_point(
                    point, np.stack([pixels_a[i], pixels_b[i]]),
                    np.stack([ka, kb]), np.stack([pose_a, pose_b]),
                    limit_px=max_error,
                )
                points[i] = fitted.point
        uv_a, z_a = project(points, ka, pose_a)
        uv_b, z_b = project(points, kb, pose_b)
        error = np.stack(
            [
                np.linalg.norm(uv_a - pixels_a, axis=1),
                np.linalg.norm(uv_b - pixels_b, axis=1),
            ],
            axis=1,
        )
        ca = -pose_a[:3, :3].T @ pose_a[:3, 3]
        cb = -pose_b[:3, :3].T @ pose_b[:3, 3]
        rays_a = points - ca
        rays_b = points - cb
        rays_a /= np.linalg.norm(rays_a, axis=1)[:, None]
        rays_b /= np.linalg.norm(rays_b, axis=1)[:, None]
        angles = np.degrees(np.arccos(np.clip(np.sum(rays_a * rays_b, axis=1), -1, 1)))
    valid = (
        np.isfinite(points).all(axis=1)
        & np.isfinite(error).all(axis=1)
        & np.isfinite(angles)
        & (z_a > 0)
        & (z_b > 0)
        & (error.max(axis=1) < max_error)
        & (angles >= min_angle)
    )
    if np.linalg.norm(ca - cb) < 1e-9:
        valid[:] = False
    return TriangulationResult(points, valid, error, angles)


@dataclass
class ObjectFeatures:
    observation: Observation
    pixels: Array
    descriptors: NDArray[np.float32] | None
    intrinsics: Array
    pose: Array
    gray: NDArray[np.uint8]
    mask: NDArray[np.uint8]


def masked_flow_correspondences(
    a: ObjectFeatures, b: ObjectFeatures
) -> tuple[Array, Array]:
    """Native-resolution corner flow checked forward/backward and against both masks.

    Returns only actual tracked image points; no centroid or box-center matches
    are manufactured. Photometric NCC rejects low-texture or mismatched patches.
    """
    empty = np.empty((0, 2), float)
    if a.gray.shape != b.gray.shape:
        return empty, empty
    corners = cv2.goodFeaturesToTrack(a.gray, 100, 0.01, 3, mask=a.mask, blockSize=3)
    if corners is None:
        return empty, empty
    forward, status, _ = cv2.calcOpticalFlowPyrLK(
        a.gray, b.gray, corners, None, winSize=(21, 21), maxLevel=3
    )
    if forward is None or status is None:
        return empty, empty
    backward, back_status, _ = cv2.calcOpticalFlowPyrLK(
        b.gray, a.gray, forward, None, winSize=(21, 21), maxLevel=3
    )
    if backward is None or back_status is None:
        return empty, empty
    valid = (status[:, 0] > 0) & (back_status[:, 0] > 0)
    valid &= np.linalg.norm(backward[:, 0] - corners[:, 0], axis=1) <= 1.0
    pa, pb = [], []
    h, w = a.gray.shape
    for point_a, point_b, supported in zip(corners[:, 0], forward[:, 0], valid):
        if not supported or not np.isfinite(point_b).all():
            continue
        x, y = np.rint(point_b).astype(int)
        if not 3 <= x < w - 3 or not 3 <= y < h - 3 or not b.mask[y, x]:
            continue
        xa, ya = np.rint(point_a).astype(int)
        if not 3 <= xa < w - 3 or not 3 <= ya < h - 3:
            continue
        patch_a = cv2.getRectSubPix(a.gray, (7, 7), tuple(map(float, point_a))).astype(
            float
        )
        patch_b = cv2.getRectSubPix(b.gray, (7, 7), tuple(map(float, point_b))).astype(
            float
        )
        patch_a -= patch_a.mean()
        patch_b -= patch_b.mean()
        length = np.linalg.norm(patch_a) * np.linalg.norm(patch_b)
        if length <= 1e-6 or float(np.sum(patch_a * patch_b) / length) < 0.85:
            continue
        pa.append(point_a)
        pb.append(point_b)
    return np.asarray(pa, float).reshape(-1, 2), np.asarray(pb, float).reshape(-1, 2)


def recover_object_geometry(
    observations: Sequence[Observation],
    reconstruction: Reconstruction,
    frames: Sequence[Frame],
    output: Path,
    config: Config,
) -> list[Observation]:
    """Cross-check masked features across views, rejecting competing object links.

    Native mask/image coordinates are preserved. A proposed pair needs semantic
    compatibility, ReID similarity, multiple matched features, geometric
    agreement and an unambiguous best candidate on BOTH sides. Existing neural
    geometry is replaced only for previously unlocalized observations; measured
    sensor geometry is never replaced. Every attempted candidate is audited.
    """
    lookup = {f.id: f for f in frames}
    features = []
    sift = cv2.SIFT_create(nfeatures=1200, contrastThreshold=0.02)
    for observation in observations:
        fid = observation.frame_id
        frame = lookup[fid]
        image = cv2.imread(frame.path)
        mask = cv2.imread(
            str(
                output
                / "evidence/supporting_frames"
                / observation.provenance["mask_file"]
            ),
            cv2.IMREAD_GRAYSCALE,
        )
        if image is None or mask is None or mask.shape != image.shape[:2]:
            raise ValueError(
                "Object geometry requires matching native image and mask coordinates"
            )
        surface = reconstruction.surfaces.get(fid)
        if surface is None:
            continue
        geometry_shape = surface["depth"].shape
        k = np.asarray(surface["intrinsics"], float)
        if config.geometry_backend == "vggt" and not config.metric_geometry_enabled:
            transform = native_to_geometry_matrix(
                image.shape, geometry_shape, config.neural_preprocessing
            )
            k = np.linalg.inv(transform) @ k
        # Erosion avoids using background pixels along uncertain mask edges.
        interior = cv2.erode(mask, np.ones((3, 3), np.uint8))
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        keypoints, descriptors = sift.detectAndCompute(gray, interior)
        pixels = np.asarray([point.pt for point in keypoints], float).reshape(-1, 2)
        features.append(
            ObjectFeatures(
                observation,
                pixels,
                descriptors,
                k,
                np.asarray(reconstruction.poses[fid], float),
                gray,
                interior,
            )
        )
    audit: list[dict[str, Any]] = []
    candidates = []
    gray_cache: dict[str, NDArray[np.uint8]] = {}
    learned_matcher = None
    learned_pairs = 0
    if config.learned_matching_checkpoint:
        from .learned_matching import LearnedCropMatcher
        learned_matcher = LearnedCropMatcher(Path(config.learned_matching_checkpoint))
    for i, a in enumerate(features):
        for j in range(i + 1, len(features)):
            b = features[j]
            if a.observation.frame_id == b.observation.frame_id:
                continue
            if semantic_label(a.observation.label) != semantic_label(
                b.observation.label
            ):
                raw_a = a.observation.provenance.get(
                    "raw_detector_label", a.observation.label
                )
                raw_b = b.observation.provenance.get(
                    "raw_detector_label", b.observation.label
                )
                if "unidentified object" not in (
                    a.observation.label,
                    b.observation.label,
                ) or semantic_label(raw_a) != semantic_label(raw_b):
                    continue
            row = {
                "before_id": a.observation.id,
                "after_id": b.observation.id,
                "gate": None,
                "calibrated": False,
            }
            audit.append(row)
            try:
                va = unit_descriptor(a.observation.descriptor)
                vb = unit_descriptor(b.observation.descriptor)
                cosine = float(va @ vb) if va.shape == vb.shape else -1.0
            except (ValueError, TypeError):
                cosine = -1.0
            row["appearance_cosine"] = cosine
            trace_a, trace_b = np.empty((0, 2)), np.empty((0, 2))
            pairs = match(a.descriptors, b.descriptors)
            if (
                config.temporal_geometry_trace_enabled
                and cosine >= 0.75
                and (cosine < config.identity_cosine or len(pairs) < 8)
            ):
                if a.observation.timestamp < b.observation.timestamp:
                    trace_a, trace_b = temporal_masked_correspondences(
                        a, b, frames, gray_cache
                    )
                else:
                    trace_b, trace_a = temporal_masked_correspondences(
                        b, a, frames, gray_cache
                    )
            row["temporal_trace_points"] = len(trace_a)
            if cosine < config.identity_cosine:
                if len(trace_a) < 8:
                    row["gate"] = "appearance_gate"
                    continue
            row["feature_matches"] = len(pairs)
            if pairs:
                ia, ib = np.asarray(pairs).T
                pixels_a, pixels_b = a.pixels[ia], b.pixels[ib]
            else:
                pixels_a, pixels_b = np.empty((0, 2)), np.empty((0, 2))
            flow_a, flow_b = masked_flow_correspondences(a, b)
            for point_a, point_b in zip(flow_a, flow_b):
                if (
                    len(pixels_a)
                    and np.min(np.linalg.norm(pixels_a - point_a, axis=1)) < 2
                ):
                    continue
                pixels_a = np.vstack([pixels_a, point_a])
                pixels_b = np.vstack([pixels_b, point_b])
            row["checked_flow_correspondences"] = len(pixels_a) - len(pairs)
            for point_a, point_b in zip(trace_a, trace_b):
                if (
                    len(pixels_a)
                    and np.min(np.linalg.norm(pixels_a - point_a, axis=1)) < 2
                ):
                    continue
                pixels_a = np.vstack([pixels_a, point_a])
                pixels_b = np.vstack([pixels_b, point_b])
            if len(pixels_a) < config.triangulation_min_points and learned_matcher is None:
                row["gate"] = "insufficient_masked_correspondences"
                continue
            result = triangulate_checked(
                pixels_a,
                pixels_b,
                a.intrinsics,
                b.intrinsics,
                a.pose,
                b.pose,
                config.triangulation_max_reprojection_px,
                config.triangulation_min_degrees,
            )
            count = int(result.valid.sum())
            small = max(np.count_nonzero(a.mask)/a.mask.size,
                        np.count_nonzero(b.mask)/b.mask.size) < .02
            if (learned_matcher is not None and small and count < config.triangulation_min_points
                    and learned_pairs < config.learned_matching_max_pairs):
                learned_pairs += 1
                la,lb = learned_matcher.match(a.gray,a.mask,b.gray,b.mask)
                row['learned_matcher'] = {'model':'kornia/loftr indoor_ds_new',
                    'actual_neural_inference':True,'raw_score_gate':.8,'masked_matches':len(la)}
                if len(la) >= config.triangulation_min_points:
                    proposal = triangulate_checked(la,lb,a.intrinsics,b.intrinsics,a.pose,b.pose,
                        config.triangulation_max_reprojection_px,config.triangulation_min_degrees)
                    support_count = int(proposal.valid.sum())
                    row['learned_matcher']['supported_points'] = support_count
                    if support_count > count and support_count/len(la) >= .5:
                        result,pixels_a,pixels_b,count = proposal,la,lb,support_count
                        row['learned_matcher']['selected'] = True
            row["supported_points"] = count
            row["supported_fraction"] = count / max(1, len(pixels_a))
            row["reprojection_errors_px"] = [
                [float(e) if np.isfinite(e) else None for e in pair]
                for pair in result.reprojection_errors
            ]
            row["maximum_accepted_reprojection_px"] = (
                float(result.reprojection_errors[result.valid].max()) if count else None
            )
            row["optimization"] = "fixed-camera point bundle adjustment; Huber loss; strict final native-pixel gate"
            finite_errors = result.reprojection_errors[
                np.isfinite(result.reprojection_errors)
            ]
            finite_angles = result.angles_degrees[np.isfinite(result.angles_degrees)]
            row["median_candidate_reprojection_px"] = (
                float(np.median(finite_errors)) if len(finite_errors) else None
            )
            row["maximum_candidate_parallax_degrees"] = (
                float(finite_angles.max()) if len(finite_angles) else None
            )
            if count < config.triangulation_min_points or count / len(pixels_a) < 0.5:
                row["gate"] = "reprojection_cheirality_or_parallax_gate"
                continue
            if cosine < config.identity_cosine and (
                count < 8 or count / len(pixels_a) < 0.75
            ):
                row["gate"] = "insufficient_strong_temporal_geometry"
                continue
            row["temporal_trace_validated"] = (
                len(trace_a) >= 8 and count >= 8 and count / len(pixels_a) >= 0.75
            )
            row["median_reprojection_px"] = float(
                np.median(result.reprojection_errors[result.valid])
            )
            row["minimum_parallax_degrees"] = float(
                result.angles_degrees[result.valid].min()
            )
            score = count * cosine
            candidates.append((i, j, score, result, row))
    accepted = []
    for i, j, score, result, row in candidates:
        af, bf = features[i].observation.frame_id, features[j].observation.frame_id
        competitors = [
            s
            for x, y, s, _, _ in candidates
            if (x, y) != (i, j)
            and (
                (x == i and features[y].observation.frame_id == bf)
                or (y == j and features[x].observation.frame_id == af)
                or (y == i and features[x].observation.frame_id == bf)
                or (x == j and features[y].observation.frame_id == af)
            )
        ]
        if competitors and score <= max(competitors) * 1.1:
            row["gate"] = "competing_masked_object_correspondences"
            continue
        row["accepted"] = True
        accepted.append((i, j, result, row))
    support: dict[str, list[tuple[Array, dict[str, Any]]]] = {}
    for i, j, result, row in accepted:
        for index in [i, j]:
            support.setdefault(features[index].observation.id, []).append(
                (result.points[result.valid], row)
            )
    recovered = []
    for observation in observations:
        if observation.id not in support:
            recovered.append(observation)
            continue
        point_sets, rows = zip(*support[observation.id])
        points = np.concatenate(point_sets)
        provenance = dict(observation.provenance)
        provenance["multiview_geometry"] = {
            "source": "Masked SIFT/checked optical flow + registered camera-ray triangulation with Huber refinement",
            "pose_calibration_verified": False,
            "points": len(points),
            "candidate_pairs": list(rows),
            "physical_change_verified": False,
        }
        if observation.position is None and not config.metric_geometry_enabled:
            position = np.median(points, axis=0).tolist()
            extent = np.quantile(points, [0.05, 0.95], axis=0)
            provenance["extent"] = (
                extent.tolist() if np.all(extent[1] - extent[0] > 1e-9) else None
            )
            provenance["location_point_count"] = len(points)
            provenance["localization"] = {
                "source": "reprojection-validated object correspondences",
                "geometry_source": "triangulated camera rays; predicted calibration",
                "independent_geometry_validation": True,
                "metric": False,
            }
            recovered.append(
                observation.model_copy(
                    update={"position": position, "provenance": provenance}
                )
            )
        else:
            recovered.append(observation.model_copy(update={"provenance": provenance}))
    directory = output / "diagnostics"
    directory.mkdir(exist_ok=True)
    import json

    (
        directory
        / f"{observations[0].provenance.get('state', 'scene') if observations else 'scene'}_object_geometry.json"
    ).write_text(
        json.dumps(
            {
                "candidates": audit,
                "accepted_pairs": len(accepted),
                "recovered_objects": sum(
                    a.position is None and b.position is not None
                    for a, b in zip(observations, recovered)
                ),
                "calibrated": False,
                "limitations": [
                    "Predicted camera poses and intrinsics remain uncertain",
                    "No metric scale inferred",
                ],
            },
            indent=2,
        )
    )
    return recovered


def geometric_view_support(
    observations: Sequence[Observation], position: list[float] | None, config: Config
) -> dict[str, Any]:
    """Count distinct rays only for actual accepted correspondence pairs in a track.

    A camera-angle threshold alone is not proof of useful observations. This
    alternative needs positive-depth, reprojection-checked features and angular
    parallax. It does not call nearby duplicate images independent evidence.
    """
    if (
        not getattr(config, "geometry_supported_views_enabled", False)
        or position is None
    ):
        return {"supported_views": 0, "observation_ids": [], "calibrated": False}
    ids = {o.id for o in observations}
    supported: set[str] = set()
    for observation in observations:
        pairs = observation.provenance.get("multiview_geometry", {}).get(
            "candidate_pairs", []
        )
        for pair in pairs:
            a, b = pair["before_id"], pair["after_id"]
            if (
                a in ids
                and b in ids
                and pair.get("accepted")
                and pair.get("gate") is None
            ):
                supported.update([a, b])
    from .semantics import independent_views

    selected = [o for o in observations if o.id in supported]
    return {
        "supported_views": independent_views(
            selected, position, config.triangulation_min_degrees
        ),
        "observation_ids": sorted(supported),
        "minimum_angle_degrees": config.triangulation_min_degrees,
        "source": "Correspondence-validated distinct camera rays; not statistical independence",
        "calibrated": False,
    }


def temporal_masked_correspondences(
    a: ObjectFeatures,
    b: ObjectFeatures,
    frames: Sequence[Frame],
    gray_cache: dict[str, NDArray[np.uint8]],
) -> tuple[Array, Array]:
    """Carry real corners through intermediate observations, with per-hop FB/NCC.

    No object centers are invented. At least two intermediate sampled frames
    are required. A lost/occluded corner is never resurrected onto another item.
    The final observed mask must contain every retained endpoint.
    """
    empty = np.empty((0, 2), float)
    between = sorted(
        [
            f
            for f in frames
            if a.observation.timestamp < f.timestamp < b.observation.timestamp
        ],
        key=lambda f: f.timestamp,
    )
    if len(between) < 2 or a.gray.shape != b.gray.shape:
        return empty, empty
    corners = cv2.goodFeaturesToTrack(a.gray, 100, 0.01, 3, mask=a.mask, blockSize=3)
    if corners is None:
        return empty, empty
    start = corners[:, 0].copy()
    current = corners.copy()
    previous = a.gray
    for frame in [*between, None]:
        if frame is None:
            image = b.gray
        else:
            if frame.id not in gray_cache:
                image = cv2.imread(frame.path, cv2.IMREAD_GRAYSCALE)
                if image is None:
                    raise ValueError("Cannot decode temporal geometry frame")
                gray_cache[frame.id] = image
            image = gray_cache[frame.id]
        if image.shape != previous.shape:
            return empty, empty
        forward, status, _ = cv2.calcOpticalFlowPyrLK(
            previous, image, current, None, winSize=(21, 21), maxLevel=3
        )
        if forward is None:
            return empty, empty
        backward, back_status, _ = cv2.calcOpticalFlowPyrLK(
            image, previous, forward, None, winSize=(21, 21), maxLevel=3
        )
        if backward is None:
            return empty, empty
        valid = (
            (status[:, 0] > 0)
            & (back_status[:, 0] > 0)
            & (np.linalg.norm(backward[:, 0] - current[:, 0], axis=1) <= 1.0)
        )
        h, w = image.shape
        for i, (pa, pb) in enumerate(zip(current[:, 0], forward[:, 0])):
            if not valid[i] or not np.isfinite(pb).all():
                valid[i] = False
                continue
            x, y = np.rint(pb).astype(int)
            xa, ya = np.rint(pa).astype(int)
            if (
                not 3 <= x < w - 3
                or not 3 <= y < h - 3
                or not 3 <= xa < w - 3
                or not 3 <= ya < h - 3
            ):
                valid[i] = False
                continue
            first = cv2.getRectSubPix(previous, (7, 7), tuple(map(float, pa))).astype(
                float
            )
            second = cv2.getRectSubPix(image, (7, 7), tuple(map(float, pb))).astype(
                float
            )
            first -= first.mean()
            second -= second.mean()
            length = np.linalg.norm(first) * np.linalg.norm(second)
            if length <= 1e-6 or float(np.sum(first * second) / length) < 0.9:
                valid[i] = False
        start = start[valid]
        current = forward[valid]
        if len(start) < 8:
            return empty, empty
        previous = image
    x, y = np.rint(current[:, 0]).astype(int).T
    valid = b.mask[y, x] > 0
    return start[valid].astype(float), current[valid, 0].astype(float)
