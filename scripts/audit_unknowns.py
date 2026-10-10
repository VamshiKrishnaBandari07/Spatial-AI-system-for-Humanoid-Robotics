"""Audit saved real decisions without inferring unsupported causes or labels."""

import argparse
import csv
import json
from collections import Counter
from pathlib import Path
import cv2
import numpy as np
from spatialmemory.benchmarking import sha256
from spatialmemory.contracts import Config
from spatialmemory.memory import Memory
from spatialmemory.semantics import group_objects


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run", type=Path, required=True)
    p.add_argument("--output", type=Path, default=Path("reports"))
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    report = json.loads((args.run / "report.json").read_text())
    memory = Memory(args.run / "spatial_memory.sqlite")
    try:
        observations = sorted(memory.observations(), key=lambda o: (o.timestamp, o.id))
    finally:
        memory.close()
    cfg = Config.model_validate(report["config"])
    groups = {
        state: group_objects(
            [o for o in observations if o.provenance["state"] == state], state, cfg
        )
        for state in ["before", "after"]
    }
    lookup = {o.id: (state, o) for state, objects in groups.items() for o in objects}
    truth = json.loads(
        (args.run / "visualization/evaluation_manifest.json").read_text()
    )
    rows = []
    mask_rows = []
    norms = []
    for o in observations:
        v = np.asarray(o.descriptor, float)
        norms.append(float(np.linalg.norm(v)))
        mask = cv2.imread(
            str(args.run / "evidence/supporting_frames" / o.provenance["mask_file"]),
            cv2.IMREAD_GRAYSCALE,
        )
        w, h = o.provenance["image_size"]
        x0, y0, x1, y1 = o.provenance["box"]
        valid = mask is not None and mask.shape == (h, w)
        area = int((mask > 0).sum()) if valid else None
        outside = None
        if valid:
            outside = int(area - (mask[y0:y1, x0:x1] > 0).sum())
        mask_rows.append(
            {
                "observation_id": o.id,
                "dimensions_valid": valid,
                "outside_box_pixels": outside,
                "retained_pixels": area,
                "box_fill_fraction": area / max((x1 - x0) * (y1 - y0), 1)
                if valid
                else None,
                "raw_predicted_iou": o.provenance["segmentation"]["raw_predicted_iou"],
                "quality_is_calibrated": False,
                "finite_descriptor": bool(np.isfinite(v).all()),
                "descriptor_norm": float(np.linalg.norm(v)),
                "appearance_revision": o.provenance["appearance"]["revision"],
            }
        )
    mask_lookup = {r["observation_id"]: r for r in mask_rows}
    for e in report["events"]:
        if e["change"] != "UNKNOWN":
            continue
        evidence = e["evidence"]
        candidates = []
        secondary = []
        gt = []
        if e.get("object_id"):
            state, obj = lookup[e["object_id"]]
            other = groups["after" if state == "before" else "before"]
            for b in other:
                va, vb = np.asarray(obj.descriptor), np.asarray(b.descriptor)
                cosine = float(
                    va @ vb / (np.linalg.norm(va) * np.linalg.norm(vb) + 1e-12)
                )
                distance = (
                    float(np.linalg.norm(np.asarray(obj.position) - b.position))
                    if obj.position is not None and b.position is not None
                    else None
                )
                candidates.append(
                    {
                        "object_id": b.id,
                        "label": b.label,
                        "exact_semantic_match": obj.label == b.label,
                        "appearance_cosine": cosine,
                        "passes_original_appearance_gate": cosine
                        >= cfg.identity_cosine,
                        "registered_centroid_distance": distance,
                        "geometric_compatibility": "distance only; no verified identity or calibrated spatial gate",
                        "independent_views": b.distinct_views,
                    }
                )
            same = [c for c in candidates if c["exact_semantic_match"]]
            primary = (
                "no_exact_semantic_candidate"
                if not same
                else "appearance_gate_rejected_all_candidates"
            )
            if obj.position is None:
                secondary.append("missing_localization")
            if obj.distinct_views < cfg.min_object_views:
                secondary.append("insufficient_independent_views")
            if (
                evidence.get("visibility", {}).get("supported_free_views", 0)
                < cfg.min_object_views
            ):
                secondary.append("insufficient_positive_free_space_views")
            for t in truth["objects"]:
                if t["change_type"] == "UNKNOWN":
                    continue
                label = lambda x: {"trash can": "trash"}.get(x, x)
                distance = (
                    float(
                        np.linalg.norm(np.asarray(obj.position) - t["before_position"])
                    )
                    if obj.position is not None and t.get("before_position") is not None
                    else None
                )
                if label(obj.label) == label(t["label"]):
                    gt.append(
                        {
                            "native_object_id": t["object_id"],
                            "supported_class": t["change_type"],
                            "distance_metres": distance,
                            "unique_link_verified": False,
                            "reason": "label alone is not identity; native audit is not independent human review",
                        }
                    )
            views = obj.distinct_views
            masks = [mask_lookup[o.id] for o in obj.observations]
            local = {
                "point_counts": [
                    o.provenance["location_point_count"] for o in obj.observations
                ],
                "observed_centroid_spread": obj.position_spread,
                "covariance_calibrated": False,
            }
        else:
            primary = "geometric_surface_difference_without_positive_free_space"
            views = None
            masks = []
            local = {
                "point_count": evidence.get("point_count"),
                "object_identity_available": False,
            }
        rows.append(
            {
                "event_id": e["id"],
                "object_id": e.get("object_id"),
                "semantic_label": e["semantic_label"],
                "candidate_matches": candidates,
                "independent_views": views,
                "mask_quality": masks,
                "registration": {
                    "valid": report["registration"]["valid"],
                    "rmse_metres": report["registration"]["rmse"],
                    "heldout_rmse_metres": report["registration"]["heldout_rmse"],
                    "covariance_calibrated": False,
                },
                "localization": local,
                "visibility": {
                    "supported_free_views": evidence.get("visibility", {}).get(
                        "supported_free_views", 0
                    ),
                    "view_state_counts": dict(
                        Counter(
                            s
                            for v in evidence.get("visibility", {}).get("views", [])
                            for s in v["states"]
                        )
                    ),
                    "positive_free_space_evidence": evidence.get(
                        "positive_free_space_evidence"
                    ),
                },
                "primary_blocking_gate": primary,
                "secondary_blocking_gates": secondary,
                "reviewed_ground_truth_candidates": gt,
                "ground_truth_identity_verified": False,
            }
        )
    result = {
        "source_run": str(args.run),
        "report_sha256": sha256(args.run / "report.json"),
        "unknown_count": len(rows),
        "decisions": rows,
        "embedding_audit": {
            "observation_count": len(norms),
            "norm_min": min(norms),
            "norm_max": max(norms),
            "mask_coordinate_checks_passed": all(
                r["dimensions_valid"] and r["outside_box_pixels"] == 0
                for r in mask_rows
            ),
            "independently_reviewed_same_different_object_distributions": None,
            "reason": "No independent human-reviewed observation-to-native identity set; do not tune appearance threshold from these predictions.",
        },
    }
    (args.output / "unknown_diagnostics.json").write_text(
        json.dumps(result, indent=2) + "\n"
    )
    fields = [
        "event_id",
        "object_id",
        "semantic_label",
        "independent_views",
        "primary_blocking_gate",
        "secondary_blocking_gates",
        "ground_truth_identity_verified",
    ]
    with (args.output / "unknown_diagnostics.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows({k: r[k] for k in fields} for r in rows)
    counts = Counter(r["primary_blocking_gate"] for r in rows)
    summary = [
        "# UNKNOWN diagnostic audit",
        "",
        f"Saved actual run: {args.run}; {len(rows)} UNKNOWN decisions.",
        "",
        *[f"- {k}: {v}" for k, v in counts.items()],
        "",
        "Primary gates describe executed decision paths; they do not establish detector error, faulty embeddings or true object identity. Mask confidence is not true IoU. Localization and registration covariance are not calibrated. Independent reviewed matching distributions and full-scene accuracy are unavailable.",
        "",
        f"Descriptor norms: {min(norms):.8f}–{max(norms):.8f}. Mask dimensions and clipping: {result['embedding_audit']['mask_coordinate_checks_passed']}. Black-background crop and central image cropping are potential sensitivity sources, not established causes.",
    ]
    (args.output / "unknown_root_cause_summary.md").write_text(
        "\n".join(summary) + "\n"
    )
    print(
        json.dumps(
            {
                "unknown": len(rows),
                "primary_gates": dict(counts),
                "audit": result["embedding_audit"],
            }
        )
    )


if __name__ == "__main__":
    main()
