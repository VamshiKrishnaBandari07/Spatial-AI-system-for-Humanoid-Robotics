import numpy as np
from scipy.spatial import cKDTree
from .contracts import Event, Change


def clusters(points, radius, min_size):
    if not len(points):
        return []
    tree = cKDTree(points)
    remaining = set(range(len(points)))
    groups = []
    while remaining:
        seed = remaining.pop()
        group = [seed]
        queue = [seed]
        while queue:
            i = queue.pop()
            for j in tree.query_ball_point(points[i], radius):
                if j in remaining:
                    remaining.remove(j)
                    group.append(j)
                    queue.append(j)
        if len(group) >= min_size:
            groups.append(points[group])
    return groups


def compare(before, after, registration, config, free_before=None, free_after=None):
    if not registration.valid:
        raise ValueError("Registration gate failed: " + registration.reason)
    events = []
    for kind, source, target, free in [
        ("REMOVED", before, after, free_after),
        ("ADDED", after, before, free_before),
    ]:
        distance = cKDTree(target).query(source)[0]
        for group in clusters(
            source[distance > config.voxel_size * 2],
            config.voxel_size * 3,
            config.min_cluster,
        ):
            center = group.mean(0).tolist()
            observed_free = free is not None and _positive(free(group))
            change = Change(kind) if observed_free else Change.UNKNOWN
            events.append(
                Event(
                    id=f"region-{len(events)}",
                    change=change,
                    before_position=center if kind == "REMOVED" else None,
                    after_position=center if kind == "ADDED" else None,
                    evidence={
                        "geometric_candidate": kind,
                        "point_count": len(group),
                        "positive_free_space_evidence": observed_free,
                        "registration": registration.model_dump(),
                    },
                    explanation=f"{kind.title()} region candidate based on sparse surface difference. "
                    + (
                        "Measured free-space evidence supports this classification."
                        if observed_free
                        else "Visibility is unverified; this is UNKNOWN, not a confirmed change."
                    ),
                )
            )
    return events


def _positive(flag) -> bool:
    """Only a literal boolean True is positive evidence.

    Dicts, strings, arrays or counts are truthy in Python; letting them promote
    ADDED/REMOVED would let an audit record or a mislabeled value act as proof.
    """
    return isinstance(flag, (bool, np.bool_)) and bool(flag)


def classify_object(
    before, after, identity_verified, visible_free, threshold, uncertainty
):
    """Reference decision rule. Fails closed: invalid evidence yields UNKNOWN.

    ``None`` means "no observation of the object in that recording"; a position
    containing NaN/inf is an observation that cannot be used, which is UNKNOWN,
    never UNCHANGED or MOVED. Absence alone never proves a change: ADDED/REMOVED
    require positive free-space evidence (a literal True) at the other location.
    Invalid thresholds are caller errors and raise.
    """
    if not (np.isfinite(threshold) and threshold > 0
            and np.isfinite(uncertainty) and uncertainty >= 0):
        raise ValueError(
            "threshold must be finite and positive; uncertainty finite and nonnegative"
        )
    for position in (before, after):
        if position is not None:
            try:
                coordinates = np.asarray(position, dtype=float)
            except (ValueError, TypeError):
                return Change.UNKNOWN
            if coordinates.shape != (3,) or not np.isfinite(coordinates).all():
                return Change.UNKNOWN
    if before is None and after is None:
        return Change.UNKNOWN
    free = _positive(visible_free)
    if before is None:
        return Change.ADDED if free else Change.UNKNOWN
    if after is None:
        return Change.REMOVED if free else Change.UNKNOWN
    if not _positive(identity_verified):
        return Change.UNKNOWN
    displacement = np.linalg.norm(np.asarray(after, float) - np.asarray(before, float))
    if abs(displacement - threshold) <= uncertainty:
        return Change.UNKNOWN
    return Change.MOVED if displacement > threshold else Change.UNCHANGED
