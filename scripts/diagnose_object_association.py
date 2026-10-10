"""Explain conservative association rejection from saved actual observations."""

import argparse
import json
from pathlib import Path
import numpy as np
from spatialmemory.benchmarking import sha256
from spatialmemory.contracts import Config
from spatialmemory.memory import Memory, associate, Observation
from spatialmemory.semantics import group_objects


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = json.loads((args.run / "report.json").read_text())
    if report["status"] != "complete":
        raise ValueError("Completed inference required")
    cfg = Config.model_validate(report["config"])
    memory = Memory(args.run / "spatial_memory.sqlite")
    try:
        observations = sorted(memory.observations(), key=lambda o: (o.timestamp, o.id))
    finally:
        memory.close()
    groups = [
        group_objects(
            [o for o in observations if o.provenance["state"] == state], state, cfg
        )
        for state in ["before", "after"]
    ]
    rows = []
    for state, objects in zip(["before", "after"], groups):
        for obj in objects:
            rows.append(
                {
                    "state": state,
                    "object_id": obj.id,
                    "label": obj.label,
                    "observations": len(obj.observations),
                    "distinct_views": obj.distinct_views,
                    "localized": obj.position is not None,
                }
            )
    pairs = []
    for a in groups[0]:
        for b in groups[1]:
            if a.label != b.label or a.descriptor is None or b.descriptor is None:
                continue
            va, vb = np.asarray(a.descriptor), np.asarray(b.descriptor)
            cosine = float(va @ vb / (np.linalg.norm(va) * np.linalg.norm(vb) + 1e-12))
            pairs.append(
                {
                    "before": a.id,
                    "after": b.id,
                    "label": a.label,
                    "cosine": cosine,
                    "passes_appearance_gate": cosine >= cfg.identity_cosine,
                }
            )

    def reps(objects):
        return [
            Observation(
                id=o.id,
                frame_id="fused",
                timestamp=0,
                label=o.label,
                position=o.position,
                descriptor=o.descriptor,
                viewpoint="fused",
            )
            for o in objects
        ]

    matches = associate(
        reps(groups[0]), reps(groups[1]), max_cost=1 - cfg.identity_cosine
    )
    result = {
        "inference_report_sha256": sha256(args.run / "report.json"),
        "objects": rows,
        "same_label_pairs": pairs,
        "accepted_assignment_candidates": matches,
        "identity_cosine_gate": cfg.identity_cosine,
        "movement_threshold": cfg.voxel_size * 2,
        "scope": "Saved actual neural observations; diagnostics only. No annotation IDs, thresholds or predictions changed.",
    }
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(
        json.dumps(
            {
                "objects": len(rows),
                "same_label_pairs": len(pairs),
                "assignments": len(matches),
            }
        )
    )


if __name__ == "__main__":
    main()
