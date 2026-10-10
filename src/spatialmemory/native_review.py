"""Audit native 3RScan relations; unsupported canonical labels remain UNKNOWN."""

import json
from pathlib import Path
import numpy as np
from .benchmarking import sha256
from .public_data import valid_transform
from .rgbd_validation import transform_points


def read_ascii_instances(path):
    with Path(path).open() as stream:
        header = []
        while True:
            line = stream.readline().strip()
            if not line or len(header) > 1000:
                raise ValueError("Invalid PLY header")
            header.append(line)
            if line == "end_header":
                break
        if "format ascii 1.0" not in header:
            raise ValueError("ASCII instance PLY required")
        item = next(s for s in header if s.startswith("element vertex "))
        count = int(item.split()[-1])
        properties = []
        for line in header[header.index(item) + 1 :]:
            if line.startswith("element "):
                break
            if line.startswith("property "):
                properties.append(line.split()[-1])
        vertices = np.loadtxt(stream, max_rows=count, ndmin=2)
    if len(vertices) != count or not np.isfinite(vertices).all():
        raise ValueError("Invalid instance vertices")
    return vertices[:, [properties.index(s) for s in ["x", "y", "z"]]], vertices[
        :, properties.index("objectId")
    ].astype(int)


def review_native(plan, minimum_translation=0.03):
    relation = Path(plan["relationship_metadata_path"])
    if sha256(relation) != plan["relationship_metadata_sha256"]:
        raise ValueError("Native relationship hash mismatch")
    entry = next(
        e
        for e in json.loads(relation.read_text())
        if e["reference"] == plan["reference"]
    )
    pair = next(e for e in entry["scans"] if e["reference"] == plan["rescan"])
    T = valid_transform(np.asarray(pair["transform"]).reshape(4, 4, order="F"))
    objects, sources = {}, {"relationship": sha256(relation)}
    for state, scan in [("before", plan["reference"]), ("after", plan["rescan"])]:
        path = Path(plan["raw_destination"]) / scan / "semseg.v2.json"
        journal = json.loads(
            path.with_suffix(path.suffix + ".download.json").read_text()
        )
        if sha256(path) != journal["sha256"]:
            raise ValueError("Native semantic source hash mismatch")
        native = json.loads(path.read_text())
        if native["scan_id"] != scan:
            raise ValueError("Native semantic scan mismatch")
        objects[state] = {int(o["objectId"]): o for o in native["segGroups"]}
        if len(objects[state]) != len(native["segGroups"]):
            raise ValueError("Duplicate native object identity")
        sources[state] = sha256(path)
    rigid = {int(r["instance_reference"]): r for r in pair["rigid"]}
    reviewed = []
    for oid in sorted(set(objects["before"]) | set(objects["after"])):
        a = objects["before"].get(oid)
        r = rigid.get(oid)
        after_id = int(r["instance_rescan"]) if r else oid
        b = objects["after"].get(after_id)
        classification, reason, delta, after_position = (
            "UNKNOWN",
            "No explicit supported native canonical class",
            None,
            None,
        )
        before_position = a["obb"]["centroid"] if a else None
        if r and a and b and a["label"] == b["label"]:
            H = valid_transform(np.asarray(r["transform"]).reshape(4, 4, order="F"))
            after_position = transform_points(np.array([before_position]), T @ H)[
                0
            ].tolist()
            delta = float(np.linalg.norm(np.asarray(after_position) - before_position))
            if delta >= minimum_translation:
                classification, reason = (
                    "MOVED",
                    "Explicit native rigid correspondence with canonical-centroid translation above declared 3 cm review margin",
                )
            else:
                reason = "Native rigid change has sub-margin centroid translation; rotation-only/small-motion classes unsupported by canonical evaluator"
        elif b:
            after_position = transform_points(np.array([b["obb"]["centroid"]]), T)[
                0
            ].tolist()
        reviewed.append(
            {
                "object_id": str(oid),
                "native_before_id": oid if a else None,
                "native_after_id": after_id if b else None,
                "label": (a or b)["label"],
                "change_type": classification,
                "label_origin": "explicit native rigid transform"
                if r
                else "unsupported; not inferred from absence",
                "reason": reason,
                "before_position": before_position,
                "after_position": after_position,
                "native_centroid_translation_metres": delta,
                "correspondence_verified": bool(a and b and a["label"] == b["label"]),
                "native_rigid": bool(r),
            }
        )
    return {
        "scene_id": plan["reference"] + "__" + plan["rescan"],
        "source_hashes": sources,
        "native_split": entry["type"],
        "data_origin": "REAL",
        "review_method": "Deterministic source-integrity and transform-convention audit; no human-review identity claimed",
        "minimum_translation_metres": minimum_translation,
        "coordinate_frame": "reference_world",
        "position_unit": "metres",
        "T_reference_rescan": T.tolist(),
        "objects": reviewed,
        "supported_categories": ["MOVED"],
        "unsupported_categories": ["ADDED", "REMOVED", "UNCHANGED", "ROTATED"],
        "exhaustive_change_inventory": False,
    }
