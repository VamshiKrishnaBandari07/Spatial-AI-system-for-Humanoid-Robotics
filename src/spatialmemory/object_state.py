"""Strict visibility decisions, separate from cross-session change labels.

No detector absence or uncalibrated score establishes a physical removal.
Callers must supply verified whole-object geometry and independent-view counts.
"""


class ValidationFailureError(ValueError):
    """Evidence cannot support a verified visibility state."""


def evaluate_object_state(
    *,
    object_id,
    object_name,
    geometry_verified,
    wholly_outside_frustum,
    fully_occluded,
    identity_verified,
    position_verified,
    free_views,
    prior_object_verified,
    unresolved_identity=False,
    min_free_views=2,
):
    """Return a supported state or raise; confidence remains explicitly unavailable.

    Booleans must represent verified facts, not detector defaults. Partial
    visibility, missing depth and inconsistent evidence fail closed.
    """
    flags = (
        geometry_verified,
        wholly_outside_frustum,
        fully_occluded,
        identity_verified,
        position_verified,
        prior_object_verified,
        unresolved_identity,
    )
    if any(type(value) is not bool for value in flags):
        raise ValidationFailureError("Verified Boolean evidence is required")
    if (
        type(free_views) is not int
        or free_views < 0
        or type(min_free_views) is not int
        or min_free_views < 2
    ):
        raise ValidationFailureError(
            "At least two independent free-space views are required"
        )
    if not geometry_verified:
        raise ValidationFailureError(
            "Alignment and whole-object geometry are not verified"
        )
    present = identity_verified and position_verified
    if (
        (wholly_outside_frustum and (fully_occluded or present or free_views))
        or (fully_occluded and (present or free_views))
        or (present and free_views)
    ):
        raise ValidationFailureError(
            "Conflicting presence, frustum or raycasting evidence"
        )
    if wholly_outside_frustum:
        state, reason = (
            "OUT_OF_VIEW",
            "Verified whole-object bounds are outside the camera frustum.",
        )
    elif fully_occluded:
        state, reason = (
            "HIDDEN",
            "Verified foreground depth occludes the whole expected object.",
        )
    elif present and not unresolved_identity:
        state, reason = (
            "PRESENT",
            "Verified object identity and position agree in the current view.",
        )
    elif (
        prior_object_verified
        and free_views >= min_free_views
        and not unresolved_identity
        and not identity_verified
    ):
        state, reason = (
            "REMOVED",
            "Independent unobstructed views verify free space at the previous object location.",
        )
    else:
        raise ValidationFailureError("Insufficient identity or visibility evidence")
    return {
        "object_id": object_id,
        "object_name": object_name,
        "state": state,
        "calibrated_confidence": None,
        "confidence_status": "not_calibrated",
        "reasoning": reason,
    }
