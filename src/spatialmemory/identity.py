"""Gated 3D/ReID association and auditable temporal identity memory.

All costs are ranking distances, not calibrated probabilities. Inputs must share
one coordinate frame and spatial unit; monocular units need not be metres.
"""

from dataclasses import asdict, dataclass, field
import json
from pathlib import Path
from typing import Any, Protocol, Sequence

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import linear_sum_assignment

from .association import semantic_label, unit_descriptor
from .contracts import Config

Array = NDArray[np.float64]


class IdentityObject(Protocol):
    id: str
    label: str
    descriptor: list[float] | None
    position: list[float] | None


@dataclass(frozen=True)
class IdentitySettings:
    radius: float = 1.0
    proximity: float = 0.2
    appearance_floor: float = 0.85
    proximity_boost: float = 0.08
    class_mismatch_floor: float = 0.98
    ambiguity_margin: float = 0.05
    max_cost: float = 0.3
    geo_weight: float = 0.45
    app_weight: float = 0.45
    class_weight: float = 0.1
    iou_weight: float = 0.5
    ema_alpha: float = 0.2
    memory_batches: int = 30
    confirmation_hits: int = 2
    crowding_thresholds: bool = False

    def __post_init__(self) -> None:
        values = [v for v in vars(self).values()]
        if not np.isfinite(values).all():
            raise ValueError("Identity settings must be finite")
        if self.radius <= 0 or self.proximity <= 0:
            raise ValueError("Spatial scales must be positive")
        for name in (
            "appearance_floor",
            "proximity_boost",
            "class_mismatch_floor",
            "ambiguity_margin",
            "max_cost",
            "iou_weight",
            "ema_alpha",
        ):
            if not 0 <= getattr(self, name) <= 1:
                raise ValueError(f"{name} must lie in [0, 1]")
        weights = [self.geo_weight, self.app_weight, self.class_weight]
        if (
            min(weights) < 0
            or not np.isfinite(sum(weights))
            or sum(weights) <= 0
            or self.max_cost <= 0
        ):
            raise ValueError(
                "Nonnegative weights with positive total and cost required"
            )
        if (
            not isinstance(self.memory_batches, int)
            or self.memory_batches < 0
            or not isinstance(self.confirmation_hits, int)
            or self.confirmation_hits < 1
        ):
            raise ValueError("Invalid memory or confirmation length")


@dataclass(frozen=True)
class IdentityObservation:
    identifier: str
    label: str
    descriptor: Array
    position: Array | None
    extent: Array | None = None
    correspondence_peers: frozenset[str] = frozenset()
    member_ids: frozenset[str] = frozenset()

    def __post_init__(self) -> None:
        object.__setattr__(self, "descriptor", unit_descriptor(self.descriptor))
        if not self.member_ids:
            object.__setattr__(self, "member_ids", frozenset({self.identifier}))
        for name, shape in (("position", (3,)), ("extent", (2, 3))):
            value = getattr(self, name)
            if value is not None:
                array = np.asarray(value, dtype=float)
                if array.shape != shape or not np.isfinite(array).all():
                    raise ValueError(f"Invalid {name}")
                if name == "extent" and np.any(array[1] <= array[0]):
                    raise ValueError("3D boxes require positive size in every axis")
                object.__setattr__(self, name, array.copy())


def iou_3d(a: Array, b: Array) -> float:
    """Axis-aligned intersection over union, after conversion to the shared frame."""
    magnitude = max(float(np.abs(a).max()), float(np.abs(b).max()))
    if magnitude > 0:
        a, b = a / magnitude, b / magnitude
    intersection = float(
        np.prod(np.maximum(0, np.minimum(a[1], b[1]) - np.maximum(a[0], b[0])))
    )
    union = float(np.prod(a[1] - a[0]) + np.prod(b[1] - b[0]) - intersection)
    return intersection / union if union > 0 else 0.0


@dataclass
class AssociationResult:
    costs: Array
    links: list[tuple[int, int]]
    diagnostics: list[dict[str, Any]]
    deferred_columns: set[int]


def joint_assignment(
    tracks: Sequence[IdentityObservation],
    detections: Sequence[IdentityObservation],
    settings: IdentitySettings,
) -> AssociationResult:
    """Return an N×M cost matrix and gated one-to-one (track, detection) links.

    D_geo = blend(min(||x-y||/radius,1), 1-IoU); without boxes use distance.
    D_app = (1-cosine)/2; D_cls = 0 for normalized equal labels, else 1.
    C = sum(w_k*D_k)/sum(w_k). Invalid/gated entries are infinity.
    Ambiguity is rejected BEFORE Hungarian assignment, on both sides. Dummy
    rows/columns allow every item to remain unmatched. No low-quality match
    is forced just because a complete permutation exists.
    """
    n, m = len(tracks), len(detections)
    costs = np.full((n, m), np.inf)
    diagnostics: list[dict[str, Any]] = []
    cosines = np.full((n, m), -np.inf)
    distances = np.full((n, m), np.inf)
    for i, a in enumerate(tracks):
        for j, b in enumerate(detections):
            row: dict[str, Any] = {
                "before": i,
                "after": j,
                "gate": None,
                "assignment_status": "rejected",
                "score_is_probability": False,
            }
            diagnostics.append(row)
            if a.position is None or b.position is None:
                row["gate"] = "missing_localization"
                continue
            if a.descriptor.shape != b.descriptor.shape:
                row["gate"] = "descriptor_dimension_mismatch"
                continue
            with np.errstate(over="ignore", invalid="ignore"):
                distance = float(np.linalg.norm(a.position - b.position))
            if not np.isfinite(distance):
                row["gate"] = "invalid_spatial_distance"
                continue
            cosine = float(np.clip(a.descriptor @ b.descriptor, -1, 1))
            distances[i, j], cosines[i, j] = distance, cosine
            same = semantic_label(a.label) == semantic_label(b.label)
            traced_geometry = bool(
                a.member_ids & b.correspondence_peers
                or b.member_ids & a.correspondence_peers
            )
            proximity_distance = distance
            if settings.crowding_thresholds:
                proximity_distance = float("inf")
                for collection, reference, reference_index in (
                    (tracks, a, i),
                    (detections, b, j),
                ):
                    for neighbor_index, neighbor in enumerate(collection):
                        if (
                            neighbor_index == reference_index
                            or neighbor.position is None
                            or reference.position is None
                        ):
                            continue
                        if (
                            neighbor.descriptor.shape == reference.descriptor.shape
                            and float(neighbor.descriptor @ reference.descriptor)
                            >= settings.appearance_floor
                        ):
                            proximity_distance = min(
                                proximity_distance,
                                float(
                                    np.linalg.norm(
                                        neighbor.position - reference.position
                                    )
                                ),
                            )
            floor = min(
                1.0,
                settings.appearance_floor
                + settings.proximity_boost
                * max(0, 1 - proximity_distance / settings.proximity),
            )
            if not same:
                floor = max(floor, settings.class_mismatch_floor)
            if (
                traced_geometry
                and settings.crowding_thresholds
                and not np.isfinite(proximity_distance)
            ):
                # An isolated accepted native-image trajectory can corroborate a
                # viewpoint-sensitive neural embedding. Crowded lookalikes retain
                # their stricter floor and every ambiguity gate below.
                floor = 0.75
            row.update(
                centroid_distance=distance,
                appearance_cosine=cosine,
                appearance_floor=floor,
                semantic_compatible=same,
                corroborated_native_trajectory=traced_geometry,
            )
            if distance > settings.radius:
                row["gate"] = "outside_search_radius"
                continue
            if cosine < floor:
                row["gate"] = "dynamic_appearance_gate"
                continue
            geo = min(1.0, distance / settings.radius)
            overlap = None
            if a.extent is not None and b.extent is not None:
                overlap = iou_3d(a.extent, b.extent)
                geo = (1 - settings.iou_weight) * geo + settings.iou_weight * (
                    1 - overlap
                )
            parts = {
                "geometry": geo,
                "appearance": (1 - cosine) / 2,
                "class": 0.0 if same else 1.0,
            }
            weights = [settings.geo_weight, settings.app_weight, settings.class_weight]
            cost = float(np.dot(weights, list(parts.values())) / sum(weights))
            row.update(cost=cost, components=parts, iou_3d=overlap)
            if cost > settings.max_cost:
                row["gate"] = "joint_cost_gate"
                continue
            costs[i, j] = cost
            row["assignment_status"] = "eligible"
    eligible = costs.copy()
    deferred: set[int] = set()
    for row in diagnostics:
        i, j = row["before"], row["after"]
        if not np.isfinite(eligible[i, j]):
            continue
        row_rivals = [k for k in range(m) if k != j and np.isfinite(eligible[i, k])]
        col_rivals = [k for k in range(n) if k != i and np.isfinite(eligible[k, j])]
        competing = any(
            eligible[i, k] <= eligible[i, j] + settings.ambiguity_margin
            for k in row_rivals
        )
        competing |= any(
            eligible[k, j] <= eligible[i, j] + settings.ambiguity_margin
            for k in col_rivals
        )
        # Similar embeddings on neighboring objects are unsafe even if a slightly
        # better geometric cost would make Hungarian select one permutation.
        lookalike = any(
            detections[k].position is not None
            and np.linalg.norm(detections[k].position - detections[j].position)
            < settings.proximity
            and abs(cosines[i, k] - cosines[i, j]) <= settings.ambiguity_margin
            for k in row_rivals
        )
        lookalike |= any(
            tracks[k].position is not None
            and np.linalg.norm(tracks[k].position - tracks[i].position)
            < settings.proximity
            and abs(cosines[k, j] - cosines[i, j]) <= settings.ambiguity_margin
            for k in col_rivals
        )
        if competing or lookalike:
            costs[i, j] = np.inf
            row.update(gate="lookalike_ambiguity", assignment_status="deferred")
            deferred.add(j)
    links: list[tuple[int, int]] = []
    if n and m:
        augmented = np.full((n + m, n + m), 1e6)
        augmented[:n, :m] = np.where(np.isfinite(costs), costs, 1e6)
        unmatched = (settings.max_cost + 1e-9) / 2
        augmented[np.arange(n), m + np.arange(n)] = unmatched
        augmented[n + np.arange(m), np.arange(m)] = unmatched
        augmented[n:, m:] = 0
        rows, cols = linear_sum_assignment(augmented)
        links = [
            (int(i), int(j))
            for i, j in zip(rows, cols)
            if i < n and j < m and np.isfinite(costs[i, j])
        ]
    accepted = set(links)
    for row in diagnostics:
        if (row["before"], row["after"]) in accepted:
            row.update(gate=None, assignment_status="accepted")
    return AssociationResult(costs, links, diagnostics, deferred)


@dataclass
class GalleryTrack:
    identifier: str
    prototype: IdentityObservation
    hits: int = 1
    missed: int = 0
    observation_ids: list[str] = field(default_factory=list)
    confirmed: bool = False


class IdentityGallery:
    """Scene-scoped EMA gallery with an explicit occlusion/handoff memory buffer.

    Each update is one frame/segment in chronological order within its session.
    Session changes may restart timestamps; coordinate_frame may NEVER change.
    Ambiguous detections remain deferred, without contaminating prototypes or
    allocating duplicate global IDs. A track needs repeated supported hits
    before confirmation. Missing geometry never establishes a global identity.
    """

    def __init__(
        self, settings: IdentitySettings, coordinate_frame: str, prefix: str = "object"
    ) -> None:
        if not coordinate_frame:
            raise ValueError("Explicit coordinate frame required")
        self.settings = settings
        self.coordinate_frame = coordinate_frame
        self.prefix = prefix
        self.tracks: dict[str, GalleryTrack] = {}
        self.audit: list[dict[str, Any]] = []
        self._counter = 0
        self._session: str | None = None
        self._timestamp = -float("inf")
        self._sessions: set[str] = set()
        self._observed_ids: set[str] = set()

    def save(self, path: Path) -> None:
        """Atomically checkpoint scene-scoped identity memory, not raw video data."""
        tracks = []
        for track in self.tracks.values():
            p = track.prototype
            tracks.append(
                {
                    "identifier": track.identifier,
                    "label": p.label,
                    "descriptor": p.descriptor.tolist(),
                    "position": p.position.tolist() if p.position is not None else None,
                    "extent": p.extent.tolist() if p.extent is not None else None,
                    "hits": track.hits,
                    "missed": track.missed,
                    "observation_ids": track.observation_ids,
                    "correspondence_peers": sorted(p.correspondence_peers),
                    "member_ids": sorted(p.member_ids),
                }
            )
        payload = {
            "schema_version": "1.0",
            "settings": asdict(self.settings),
            "coordinate_frame": self.coordinate_frame,
            "prefix": self.prefix,
            "session": self._session,
            "sessions": sorted(self._sessions),
            "timestamp": self._timestamp if self._session else None,
            "counter": self._counter,
            "observed_ids": sorted(self._observed_ids),
            "tracks": tracks,
        }
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(path.suffix + ".tmp")
        temporary.write_text(json.dumps(payload, allow_nan=False, indent=2))
        temporary.replace(path)

    @classmethod
    def load(cls, path: Path) -> "IdentityGallery":
        """Restore validated prototypes and confirmation history for another recording."""
        data = json.loads(Path(path).read_text())
        if data.get("schema_version") != "1.0":
            raise ValueError("Unsupported identity gallery schema")
        gallery = cls(
            IdentitySettings(**data["settings"]),
            data["coordinate_frame"],
            data["prefix"],
        )
        gallery._session = data["session"]
        gallery._sessions = set(data["sessions"])
        gallery._timestamp = (
            float(data["timestamp"]) if gallery._session else -float("inf")
        )
        gallery._counter = data["counter"]
        gallery._observed_ids = set(data["observed_ids"])
        if gallery._session and (
            not np.isfinite(gallery._timestamp)
            or gallery._session not in gallery._sessions
        ):
            raise ValueError("Invalid gallery session history")
        for row in data["tracks"]:
            identifier = row["identifier"]
            if identifier in gallery.tracks or row["hits"] < 1 or row["missed"] < 0:
                raise ValueError("Invalid saved track history")
            if (
                not isinstance(row["hits"], int)
                or not isinstance(row["missed"], int)
                or row["hits"] != len(row["observation_ids"])
                or len(set(row["observation_ids"])) != row["hits"]
            ):
                raise ValueError("Invalid saved observation count")
            if not set(row["observation_ids"]).issubset(gallery._observed_ids):
                raise ValueError("Track observations missing from gallery history")
            prototype = IdentityObservation(
                identifier,
                row["label"],
                row["descriptor"],
                row["position"],
                row["extent"],
                frozenset(row.get("correspondence_peers", [])),
                frozenset(row.get("member_ids", row["observation_ids"])),
            )
            gallery.tracks[identifier] = GalleryTrack(
                identifier,
                prototype,
                row["hits"],
                row["missed"],
                row["observation_ids"],
                row["hits"] >= gallery.settings.confirmation_hits,
            )
        if gallery._counter != len(gallery.tracks):
            raise ValueError("Invalid identity allocation counter")
        if set(gallery.tracks) != {
            f"{gallery.prefix}_{i}" for i in range(gallery._counter)
        }:
            raise ValueError("Invalid saved identity namespace")
        return gallery

    def update(
        self,
        detections: Sequence[IdentityObservation],
        *,
        timestamp: float,
        session: str,
        coordinate_frame: str,
    ) -> dict[str, str | None]:
        if coordinate_frame != self.coordinate_frame:
            raise ValueError("Gallery requires a registered shared coordinate frame")
        if not np.isfinite(timestamp) or not session:
            raise ValueError("Finite timestamp and session required")
        if session == self._session and timestamp <= self._timestamp:
            raise ValueError("Frame timestamps must strictly increase within a session")
        if session != self._session and session in self._sessions:
            raise ValueError("Cannot revisit a completed session")
        if len({d.identifier for d in detections}) != len(detections):
            raise ValueError("Duplicate observation identifiers in one batch")
        if self._observed_ids.intersection(d.identifier for d in detections):
            raise ValueError("Repeated observation cannot confirm an identity")
        self._session, self._timestamp = session, timestamp
        self._sessions.add(session)
        self._observed_ids.update(d.identifier for d in detections)
        active = [
            t for t in self.tracks.values() if t.missed <= self.settings.memory_batches
        ]
        result = joint_assignment(
            [t.prototype for t in active], detections, self.settings
        )
        assignments: dict[str, str | None] = {d.identifier: None for d in detections}
        used: set[int] = set()
        for track in self.tracks.values():
            track.missed += 1
        for i, j in result.links:
            track, detection = active[i], detections[j]
            alpha = self.settings.ema_alpha
            descriptor = unit_descriptor(
                (1 - alpha) * track.prototype.descriptor + alpha * detection.descriptor
            )
            track.prototype = IdentityObservation(
                track.identifier,
                track.prototype.label,
                descriptor,
                detection.position,
                detection.extent,
                track.prototype.correspondence_peers | detection.correspondence_peers,
                track.prototype.member_ids | detection.member_ids,
            )
            track.hits += 1
            track.missed = 0
            track.confirmed = track.hits >= self.settings.confirmation_hits
            track.observation_ids.append(detection.identifier)
            assignments[detection.identifier] = track.identifier
            used.add(j)
        for j, detection in enumerate(detections):
            if j in used or j in result.deferred_columns or detection.position is None:
                continue
            identifier = f"{self.prefix}_{self._counter}"
            self._counter += 1
            self.tracks[identifier] = GalleryTrack(
                identifier,
                detection,
                observation_ids=[detection.identifier],
                confirmed=self.settings.confirmation_hits == 1,
            )
            assignments[detection.identifier] = identifier
        self.audit.append(
            {
                "session": session,
                "timestamp": timestamp,
                "assignments": assignments,
                "candidates": result.diagnostics,
                "calibrated": False,
            }
        )
        return assignments


def settings_from_config(config: Config) -> IdentitySettings:
    """Translate the central validated configuration without changing legacy modes."""
    return IdentitySettings(
        radius=config.association_search_radius,
        proximity=config.identity_proximity,
        appearance_floor=config.identity_cosine,
        proximity_boost=config.identity_proximity_boost,
        ambiguity_margin=config.association_ambiguity_margin,
        max_cost=config.identity_joint_max_cost,
        geo_weight=config.association_geometry_weight,
        app_weight=config.association_appearance_weight,
        class_weight=config.identity_class_weight,
        iou_weight=config.identity_iou_weight,
        ema_alpha=config.identity_ema_alpha,
        memory_batches=config.identity_memory_batches,
        confirmation_hits=config.identity_confirmation_hits,
        crowding_thresholds=config.identity_crowding_thresholds,
    )


def observation_from_object(obj: IdentityObject) -> IdentityObservation:
    """Adapt core ObjectState/Observation; missing or invalid embeddings fail closed."""
    extent = getattr(obj, "extent", None)
    if extent is None:
        extent = getattr(obj, "provenance", {}).get("extent")
    peers = set()
    for pair in (
        getattr(obj, "provenance", {})
        .get("multiview_geometry", {})
        .get("candidate_pairs", [])
    ):
        if (
            pair.get("accepted")
            and pair.get("gate") is None
            and pair.get("temporal_trace_validated")
        ):
            peers.update([pair["before_id"], pair["after_id"]])
    peers.discard(obj.id)
    return IdentityObservation(
        obj.id,
        obj.label,
        np.asarray(obj.descriptor, dtype=float),
        np.asarray(obj.position, dtype=float) if obj.position is not None else None,
        np.asarray(extent, dtype=float) if extent is not None else None,
        frozenset(peers),
    )


def match_core_objects(
    before: Sequence[IdentityObject],
    after: Sequence[IdentityObject],
    config: Config,
    within_recording: bool,
    diagnostics: list[dict[str, Any]] | None,
) -> list[dict[str, Any]]:
    """Bridge joint matching into consensus; independent-view gates remain in force."""

    def convert(
        objects: Sequence[IdentityObject],
    ) -> tuple[list[int], list[IdentityObservation]]:
        indices, converted = [], []
        for index, obj in enumerate(objects):
            if not within_recording and not getattr(obj, "identity_confirmed", True):
                continue
            if (
                not within_recording
                and getattr(obj, "distinct_views", 0) < config.min_object_views
            ):
                continue
            try:
                value = observation_from_object(obj)
            except (ValueError, TypeError):
                continue
            indices.append(index)
            converted.append(value)
        return indices, converted

    ai, a = convert(before)
    bi, b = convert(after)
    result = joint_assignment(a, b, settings_from_config(config))
    if diagnostics is not None:
        indexed = {(ai[r["before"]], bi[r["after"]]): r for r in result.diagnostics}
        for i, left in enumerate(before):
            for j, right in enumerate(after):
                row = indexed.get((i, j))
                if row is None:
                    low_views = (
                        not within_recording
                        and min(
                            getattr(left, "distinct_views", 0),
                            getattr(right, "distinct_views", 0),
                        )
                        < config.min_object_views
                    )
                    row = {
                        "gate": "insufficient_independent_views"
                        if low_views
                        else "invalid_identity_observation",
                        "assignment_status": "rejected",
                    }
                    if not within_recording and not all(
                        getattr(o, "identity_confirmed", True) for o in (left, right)
                    ):
                        row["gate"] = "unconfirmed_temporal_identity"
                diagnostics.append(
                    {
                        **row,
                        "before": i,
                        "after": j,
                        "before_id": left.id,
                        "after_id": right.id,
                        "calibrated": False,
                    }
                )
    links = []
    for i, j in result.links:
        row = next(
            r for r in result.diagnostics if r["before"] == i and r["after"] == j
        )
        links.append(
            {
                "before": ai[i],
                "after": bi[j],
                "cost": float(result.costs[i, j]),
                "identity_verified": True,
                "alternatives": [bi[j]],
                "rivals": [ai[i]],
                "method": "joint",
                "cues": row,
                "score_is_probability": False,
            }
        )
    return links
