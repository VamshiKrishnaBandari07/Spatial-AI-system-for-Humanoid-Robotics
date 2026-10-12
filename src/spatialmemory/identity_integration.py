"""Temporal gallery evidence feeding the core object-centric 3D representation."""

from collections import defaultdict
from pathlib import Path
import json
from typing import Sequence

import numpy as np

from .contracts import Config
from .memory import Observation
from .identity import IdentityGallery, observation_from_object, settings_from_config
from .semantics import ObjectState, independent_views


def group_with_gallery(
    observations: Sequence[Observation],
    state: str,
    config: Config,
    checkpoint: Path | None = None,
) -> list[ObjectState]:
    """Group chronological observations using EMA, geometry and occlusion memory.

    Local track IDs are provisional. Cross-recording global IDs are committed by
    central consensus only after its identity and independent-view gates pass.
    Unlocalized/ambiguous evidence is retained independently, never dropped or
    silently merged. Gallery audits are attached to observation provenance.
    """
    gallery = IdentityGallery(
        settings_from_config(config), "shared_world", state + "_track"
    )
    batches: dict[str, list[Observation]] = defaultdict(list)
    for observation in observations:
        batches[observation.frame_id].append(observation)
    grouped: dict[str, list[Observation]] = defaultdict(list)
    for batch in sorted(batches.values(), key=lambda rows: rows[0].timestamp):
        converted = []
        for observation in batch:
            try:
                converted.append(observation_from_object(observation))
            except (ValueError, TypeError):
                pass
        assignments = gallery.update(
            converted,
            timestamp=batch[0].timestamp,
            session=state,
            coordinate_frame="shared_world",
        )
        audit = gallery.audit[-1]
        for observation in batch:
            identifier = assignments.get(observation.id)
            provenance = dict(observation.provenance)
            provenance["identity_gallery"] = {
                "track_id": identifier,
                "status": "provisional" if identifier else "deferred",
                "calibrated": False,
                "candidates": [
                    row
                    for row in audit["candidates"]
                    if converted[row["after"]].identifier == observation.id
                ],
            }
            # The observation ID distinguishes unresolved same-frame lookalikes.
            key = identifier or f"{state}_unresolved_{observation.id}"
            grouped[key].append(
                observation.model_copy(update={"provenance": provenance})
            )
    result = []
    for identifier, rows in grouped.items():
        track = gallery.tracks.get(identifier)
        positions = [row.position for row in rows if row.position is not None]
        position = np.median(positions, axis=0).tolist() if positions else None
        descriptor = (
            track.prototype.descriptor.tolist() if track else rows[-1].descriptor
        )
        for row in rows:
            row.provenance["identity_gallery"]["status"] = (
                "confirmed_local"
                if track and track.confirmed
                else "provisional"
                if track
                else "deferred"
            )
        obj = ObjectState(
            identifier,
            track.prototype.label if track else rows[0].label,
            position,
            descriptor,
            rows,
            track.prototype.extent.tolist()
            if track and track.prototype.extent is not None
            else rows[0].provenance.get("extent"),
        )
        obj.distinct_views = independent_views(rows, position, config.viewpoint_degrees)
        if getattr(config, "geometry_supported_views_enabled", False):
            from .object_geometry import geometric_view_support

            view_support = geometric_view_support(rows, position, config)
            obj.distinct_views = max(
                obj.distinct_views, view_support["supported_views"]
            )
            for row in rows:
                row.provenance["geometric_view_support"] = view_support
        obj.identity_confirmed = bool(track and track.confirmed)
        if positions:
            obj.position_spread = float(
                np.max(np.linalg.norm(np.asarray(positions) - position, axis=1))
            )
        result.append(obj)
    if checkpoint is not None:
        gallery.save(checkpoint)
    return result


def save_identity_audit(
    output: Path, groups: Sequence[ObjectState], state: str
) -> None:
    """Persist local support separately from physical change predictions."""
    path = output / "diagnostics" / f"{state}_identity_gallery.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = [
        {
            "object_id": group.id,
            "distinct_views": group.distinct_views,
            "observations": [
                {
                    "id": o.id,
                    "frame_id": o.frame_id,
                    **o.provenance.get("identity_gallery", {}),
                }
                for o in group.observations
            ],
        }
        for group in groups
    ]
    path.write_text(
        json.dumps(
            {"calibrated": False, "physical_change_verified": False, "tracks": rows},
            indent=2,
        )
    )
