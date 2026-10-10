"""Descriptive matcher comparison on identical saved neural observations."""

import argparse
from dataclasses import replace
import json
from pathlib import Path
from spatialmemory.association import match_objects
from spatialmemory.benchmarking import sha256
from spatialmemory.contracts import Config
from spatialmemory.memory import Memory
from spatialmemory.movement import bounded_movement
from spatialmemory.semantics import group_objects


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    report = json.loads((args.run / "report.json").read_text())
    cfg = Config.model_validate(report["config"]).model_copy(
        update={"association_mode": "legacy"}
    )
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
    for mode in ["legacy", "appearance", "semantic", "semantic_geometry", "multiview"]:
        matches = match_objects(
            *groups, cfg.model_copy(update={"association_mode": mode})
        )
        rows.append(
            {
                "mode": mode,
                "assignments": len(matches),
                "identity_gate_accepted": sum(m["identity_verified"] for m in matches),
                "matching_accuracy": None,
                "false_association_rate": None,
                "matches": matches,
            }
        )
    # Clearly synthetic, independently constructed error-budget sensitivity cases.
    template = groups[0][0]
    template = replace(
        template, position=[0.0, 0.0, 0.0], distinct_views=2, position_spread=0.001
    )
    synthetic = []
    for voxel in [0.01, 0.04, 0.08]:
        config = cfg.model_copy(
            update={
                "voxel_size": voxel,
                "movement_mode": "bounded",
                "movement_depth_error_bound": 0.002,
                "movement_registration_error_bound": 0.005,
                "movement_error_source": "synthetic declared exact budget; no real calibration",
            }
        )
        for d in [0, 0.005, 0.01, 0.02, 0.04, 0.08, 0.16, 0.32]:
            outcome, evidence = bounded_movement(
                template, replace(template, position=[d, 0, 0]), config
            )
            synthetic.append(
                {
                    "origin": "SYNTHETIC",
                    "voxel_size": voxel,
                    "displacement": d,
                    "prediction": outcome,
                    "interval": evidence["interval"],
                }
            )
    result = {
        "source_report_sha256": sha256(args.run / "report.json"),
        "baseline_grouping": "Identical legacy grouped observations across all matchers",
        "appearance_floor_unchanged": cfg.identity_cosine,
        "normalization": "Cosine maps [-1,1] to [0,1]; Gaussian spatial ranking within configured radius; extent-diagonal ratio; distinct-view support. Ranking scores are not probabilities.",
        "rows": rows,
        "independent_reviewed_matching_examples": 0,
        "reason": "Native object-change audit does not supply independent observation identity labels; no matching accuracy or false-association rate claimed",
        "synthetic_movement_sensitivity": synthetic,
        "legacy_note": "Original matcher already gates on exact semantic label; appearance row is the true appearance-only exploratory alternative and rejects label-conflicting verified identities",
    }
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(
        json.dumps(
            {
                "comparison": [
                    {k: r[k] for k in ["mode", "assignments", "identity_gate_accepted"]}
                    for r in rows
                ]
            }
        )
    )


if __name__ == "__main__":
    main()
