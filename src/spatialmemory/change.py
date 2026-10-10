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
            observed_free = free is not None and bool(free(group))
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


def classify_object(
    before, after, identity_verified, visible_free, threshold, uncertainty
):
    if before is None:
        return Change.ADDED if visible_free else Change.UNKNOWN
    if after is None:
        return Change.REMOVED if visible_free else Change.UNKNOWN
    if not identity_verified:
        return Change.UNKNOWN
    displacement = np.linalg.norm(np.asarray(after) - before)
    if abs(displacement - threshold) <= uncertainty:
        return Change.UNKNOWN
    return Change.MOVED if displacement > threshold else Change.UNCHANGED
