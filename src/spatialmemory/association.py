"""Configurable bounded multi-cue assignment with explicit unmatched options.

Scores are ranking affinities, not calibrated identity probabilities. The existing
appearance gate is retained; geometry never manufactures a low-appearance identity.
"""

import numpy as np
from scipy.optimize import linear_sum_assignment
from .memory import associate


def unit_descriptor(value):
    vector = np.asarray(value, float)
    if (
        vector.ndim != 1
        or not len(vector)
        or not np.isfinite(vector).all()
        or np.linalg.norm(vector) < 1e-12
    ):
        raise ValueError("Finite nonzero descriptor required")
    return vector / np.linalg.norm(vector)


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


def match_objects(before, after, config, within_recording=False):
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
            same = semantic_label(a.label) == semantic_label(b.label)
            if mode != "appearance" and not same:
                continue
            try:
                va, vb = unit_descriptor(a.descriptor), unit_descriptor(b.descriptor)
            except (ValueError, TypeError):
                continue
            if va.shape != vb.shape:
                continue
            cosine = float(np.clip(va @ vb, -1, 1))
            if cosine < config.identity_cosine:
                continue
            parts = {"appearance": (cosine + 1) / 2}
            weights = {"appearance": config.association_appearance_weight}
            distance = None
            size_ratio = None
            if mode in ["semantic_geometry", "multiview"]:
                if a.position is None or b.position is None:
                    continue
                pa, pb = np.asarray(a.position, float), np.asarray(b.position, float)
                if (
                    pa.shape != (3,)
                    or pb.shape != (3,)
                    or not np.isfinite([pa, pb]).all()
                ):
                    continue
                distance = float(np.linalg.norm(pa - pb))
                if distance > config.association_search_radius:
                    continue
                parts["geometry"] = float(
                    np.exp(-0.5 * (distance / config.association_search_radius) ** 2)
                )
                weights["geometry"] = config.association_geometry_weight
                sa, sb = extent_size(a), extent_size(b)
                if sa is None or sb is None:
                    continue
                size_ratio = min(sa, sb) / max(sa, sb)
                parts["shape"] = size_ratio
                weights["shape"] = config.association_shape_weight
            if mode == "multiview":
                views = min(
                    getattr(a, "distinct_views", 0), getattr(b, "distinct_views", 0)
                )
                if views < config.min_object_views:
                    continue
                parts["views"] = min(1.0, views / config.min_object_views)
                weights["views"] = config.association_view_weight
            score = sum(parts[k] * weights[k] for k in parts) / sum(weights.values())
            if score < config.association_min_score:
                continue
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
    return result
