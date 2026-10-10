"""Conservative displacement intervals using explicitly declared error budgets."""

import numpy as np


def bounded_movement(before, after, config):
    required = {
        "depth": config.movement_depth_error_bound,
        "registration": config.movement_registration_error_bound,
        "provenance": config.movement_error_source,
    }
    evidence = {
        "method": "Declared deterministic error budgets plus observed repeat-view spread",
        "calibrated_covariance": False,
        "posterior_probability": None,
        "error_budget": required,
        "minimum_displacement": config.voxel_size * 2,
    }
    if any(value is None for value in required.values()):
        return "UNKNOWN", {
            **evidence,
            "blocking_gate": "unquantified_depth_or_registration_error_budget",
        }
    if (
        min(before.distinct_views, after.distinct_views) < config.min_object_views
        or before.position is None
        or after.position is None
    ):
        return "UNKNOWN", {
            **evidence,
            "blocking_gate": "insufficient_localized_independent_views",
        }
    spreads = [before.position_spread, after.position_spread]
    if any(v is None or not np.isfinite(v) or v < 0 for v in spreads):
        return "UNKNOWN", {
            **evidence,
            "blocking_gate": "unmeasured_centroid_repeatability",
        }
    points = np.asarray([before.position, after.position], float)
    if points.shape != (2, 3) or not np.isfinite(points).all():
        return "UNKNOWN", {**evidence, "blocking_gate": "invalid_centroid_geometry"}
    distance = float(np.linalg.norm(points[0] - points[1]))
    radius = sum(spreads) + 2 * required["depth"] + required["registration"]
    lower, upper = max(0, distance - radius), distance + radius
    change = (
        "MOVED"
        if lower > config.voxel_size * 2
        else "UNCHANGED"
        if upper < config.voxel_size * 2
        else "UNKNOWN"
    )
    return change, {
        **evidence,
        "displacement": distance,
        "observed_spread": spreads,
        "interval": [lower, upper],
        "interval_is_statistical_confidence": False,
        "bounds_are_caller_declared_not_automatically_validated": True,
        "blocking_gate": None
        if change != "UNKNOWN"
        else "displacement_interval_crosses_effect_size",
    }
