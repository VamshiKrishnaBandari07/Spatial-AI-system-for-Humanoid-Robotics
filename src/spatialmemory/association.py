"""Configurable bounded multi-cue assignment with explicit unmatched options.

Scores are ranking affinities, not calibrated identity probabilities. The existing
appearance gate is retained; geometry never manufactures a low-appearance identity.
"""

import numpy as np
from scipy.optimize import linear_sum_assignment
from .memory import associate


def unit_descriptor(value):
    vector = np.asarray(value, float)
    if vector.ndim != 1 or not len(vector) or not np.isfinite(vector).all():
        raise ValueError("Finite nonzero descriptor required")
    # Rescale first: squaring large finite coordinates can overflow the norm.
    magnitude = float(np.max(np.abs(vector)))
    if magnitude == 0:
        raise ValueError("Finite nonzero descriptor required")
    scaled = vector / magnitude
    length = float(np.linalg.norm(scaled))
    if magnitude < 1e-12 and magnitude * length < 1e-12:
        raise ValueError("Finite nonzero descriptor required")
    return scaled / length


def semantic_label(label):
    text = " ".join(str(label).lower().strip(" .").split())
    return {"trash": "trash can", "trashcan": "trash can"}.get(text, text)


def extent_size(obj):
    extent = getattr(obj, "extent", None)
    if extent is None:
        extent = getattr(obj, "provenance", {}).get("extent")
    if extent is None:
        return None
    extent = np.asarray(extent, float)
    if (
        extent.shape != (2, 3)
        or not np.isfinite(extent).all()
        or np.any(extent[1] < extent[0])
    ):
        return None
    size = float(np.linalg.norm(extent[1] - extent[0]))
    return size if size > 1e-9 else None


def pair_measurements(a, b):
    """Observations only: raw mask scores and repeat spread are not calibration."""
    result = {
        "semantic_compatible": semantic_label(a.label) == semantic_label(b.label),
        "before_label": a.label,
        "after_label": b.label,
        "appearance_cosine": None,
        "descriptor_norms": [None, None],
        "registered_centroid_distance": None,
        "observed_extent_sizes": [extent_size(a), extent_size(b)],
        "distinct_views": [
            getattr(a, "distinct_views", 0),
            getattr(b, "distinct_views", 0),
        ],
        "position_spread": [
            getattr(a, "position_spread", None),
            getattr(b, "position_spread", None),
        ],
        "mask_quality": [],
        "calibrated": False,
    }
    for index, obj in enumerate([a, b]):
        try:
            vector = np.asarray(obj.descriptor, float)
            unit_descriptor(vector)
            with np.errstate(over="ignore"):
                norm = float(np.linalg.norm(vector))
            result["descriptor_norms"][index] = norm if np.isfinite(norm) else None
        except (ValueError, TypeError):
            pass
        result["mask_quality"].append(
            [
                {
                    "frame_id": o.frame_id,
                    "raw_predicted_iou": o.provenance.get("segmentation", {}).get(
                        "raw_predicted_iou"
                    ),
                    "area_pixels": o.provenance.get("segmentation", {}).get(
                        "area_pixels"
                    ),
                    "location_point_count": o.provenance.get("location_point_count"),
                }
                for o in getattr(obj, "observations", [])
            ]
        )
    try:
        va, vb = unit_descriptor(a.descriptor), unit_descriptor(b.descriptor)
        if va.shape == vb.shape:
            result["appearance_cosine"] = float(np.clip(va @ vb, -1, 1))
    except (ValueError, TypeError):
        pass
    if a.position is not None and b.position is not None:
        points = np.asarray([a.position, b.position], float)
        if points.shape == (2, 3) and np.isfinite(points).all():
            result["registered_centroid_distance"] = float(
                np.linalg.norm(points[0] - points[1])
            )
    return result


def match_objects(before, after, config, within_recording=False, diagnostics=None):
    mode = config.association_mode
    if mode == "legacy":
        return associate(before, after, max_cost=1 - config.identity_cosine)
    if not before or not after:
        return []
    if within_recording and mode == "multiview":
        mode = "semantic_geometry"
    n, m = len(before), len(after)
    entries = {}
    scores = np.full((n, m), -np.inf)
    for i, a in enumerate(before):
        for j, b in enumerate(after):
            row = {
                "before": i,
                "after": j,
                "before_id": a.id,
                "after_id": b.id,
                "gate": "semantic_incompatible",
                "assignment_status": "rejected",
            }
            if diagnostics is not None:
                row.update(pair_measurements(a, b))
                diagnostics.append(row)
            same = semantic_label(a.label) == semantic_label(b.label)
            if mode != "appearance" and not same:
                continue
            row["gate"] = "invalid_descriptor"
            try:
                va, vb = unit_descriptor(a.descriptor), unit_descriptor(b.descriptor)
            except (ValueError, TypeError):
                continue
            row["gate"] = "descriptor_dimension_mismatch"
            if va.shape != vb.shape:
                continue
            cosine = float(np.clip(va @ vb, -1, 1))
            row["gate"] = "appearance_below_floor"
            if cosine < config.identity_cosine:
                continue
            parts = {"appearance": (cosine + 1) / 2}
            weights = {"appearance": config.association_appearance_weight}
            distance = None
            size_ratio = None
            if mode in ["semantic_geometry", "multiview"]:
                row["gate"] = "missing_localization"
                if a.position is None or b.position is None:
                    continue
                pa, pb = np.asarray(a.position, float), np.asarray(b.position, float)
                row["gate"] = "invalid_localization"
                if (
                    pa.shape != (3,)
                    or pb.shape != (3,)
                    or not np.isfinite([pa, pb]).all()
                ):
                    continue
                distance = float(np.linalg.norm(pa - pb))
                row["gate"] = "outside_search_radius"
                if distance > config.association_search_radius:
                    continue
                parts["geometry"] = float(
                    np.exp(-0.5 * (distance / config.association_search_radius) ** 2)
                )
                weights["geometry"] = config.association_geometry_weight
                sa, sb = extent_size(a), extent_size(b)
                row["gate"] = "invalid_observed_extent"
                if sa is None or sb is None:
                    continue
                size_ratio = min(sa, sb) / max(sa, sb)
                parts["shape"] = size_ratio
                weights["shape"] = config.association_shape_weight
            if mode == "multiview":
                views = min(
                    getattr(a, "distinct_views", 0), getattr(b, "distinct_views", 0)
                )
                row["gate"] = "insufficient_independent_views"
                if views < config.min_object_views:
                    continue
                parts["views"] = min(1.0, views / config.min_object_views)
                weights["views"] = config.association_view_weight
            score = sum(parts[k] * weights[k] for k in parts) / sum(weights.values())
            row.update(
                gate="ranking_score_below_floor",
                ranking_score=score,
                score_is_probability=False,
            )
            if score < config.association_min_score:
                continue
            row.update(gate=None, assignment_status="eligible_unassigned")
            scores[i, j] = score
            entries[i, j] = {
                "appearance_cosine": cosine,
                "semantic_compatible": same,
                "centroid_distance": distance,
                "observed_extent_ratio": size_ratio,
                "components": parts,
                "weights": weights,
                "ranking_score": score,
                "score_is_probability": False,
            }
    # n+m square assignment: every object can remain unmatched on either side.
    costs = np.full((n + m, n + m), 1e6)
    costs[:n, :m] = np.where(np.isfinite(scores), 1 - scores, 1e6)
    penalty = (1 - config.association_min_score) / 2
    costs[np.arange(n), m + np.arange(n)] = penalty
    costs[n + np.arange(m), np.arange(m)] = penalty
    costs[n:, m:] = 0
    rows, cols = linear_sum_assignment(costs)
    result = []
    for i, j in zip(rows, cols):
        if i >= n or j >= m or (i, j) not in entries:
            continue
        alternatives = np.flatnonzero(
            scores[i] >= scores[i, j] - config.association_ambiguity_margin
        ).tolist()
        rivals = np.flatnonzero(
            scores[:, j] >= scores[i, j] - config.association_ambiguity_margin
        ).tolist()
        result.append(
            {
                "before": int(i),
                "after": int(j),
                "cost": float(1 - scores[i, j]),
                "alternatives": alternatives,
                "rivals": rivals,
                "identity_verified": len(alternatives) == len(rivals) == 1
                and entries[i, j]["semantic_compatible"],
                "method": mode,
                "cues": entries[i, j],
            }
        )
    if diagnostics is not None:
        assigned = {(item["before"], item["after"]): item for item in result}
        for row in diagnostics:
            item = assigned.get((row["before"], row["after"]))
            if item is not None:
                row.update(
                    assignment_status="accepted"
                    if item["identity_verified"]
                    else "ambiguous",
                    gate=None if item["identity_verified"] else "assignment_ambiguity",
                    alternatives=item["alternatives"],
                    rivals=item["rivals"],
                )
            elif row["assignment_status"] == "eligible_unassigned":
                row["gate"] = "one_to_one_assignment_or_unmatched_option"
    return result
