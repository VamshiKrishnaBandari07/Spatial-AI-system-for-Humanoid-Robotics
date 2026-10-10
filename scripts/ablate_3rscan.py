"""Replay saved neural observations; ablations do not invoke models or invent labels."""

import argparse
from collections import Counter
from dataclasses import replace
import json
from pathlib import Path

from spatialmemory.contracts import Config, Frame, Registration
from spatialmemory.inputs import load_manifest
from spatialmemory.memory import Memory
from spatialmemory.sensor_geometry import reconstruct_sensor
from spatialmemory.semantics import group_objects, consensus
from spatialmemory.benchmarking import sha256


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run", type=Path, required=True)
    p.add_argument("--manifest", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    report = json.loads((args.run / "report.json").read_text())
    if report["status"] != "complete":
        raise ValueError("Completed inference required")
    cfg = Config.model_validate(report["config"])
    _, files, _ = load_manifest(args.manifest, Path.cwd())
    reconstructions = []
    for state in ["before", "after"]:
        frames = [
            Frame.model_validate(f)
            for f in json.loads((args.run / state / "manifest.json").read_text())[
                "frames"
            ]
        ]
        reconstructions.append(reconstruct_sensor(frames, files[state]))
    memory = Memory(args.run / "spatial_memory.sqlite")
    try:
        observations = sorted(memory.observations(), key=lambda o: (o.timestamp, o.id))
    finally:
        memory.close()
    rows = []
    for mode in [
        "baseline",
        "without_multi_view_gate",
        "without_visibility",
        "without_observed_spread_band",
    ]:
        config = (
            cfg.model_copy(update={"min_object_views": 1})
            if mode == "without_multi_view_gate"
            else cfg
        )
        groups = [
            group_objects(
                [o for o in observations if o.provenance["state"] == state],
                state,
                config,
            )
            for state in ["before", "after"]
        ]
        rs = reconstructions
        registration = Registration.model_validate(report["registration"])
        if mode == "without_visibility":
            rs = [replace(r, surfaces={}) for r in rs]
        if mode == "without_observed_spread_band":
            registration = registration.model_copy(update={"rmse": 0})
            for group in groups:
                for obj in group:
                    obj.position_spread = 0
        events, matches = consensus(*groups, *rs, registration, config)
        rows.append(
            {
                "mode": mode,
                "semantic_events": len(events),
                "class_counts": dict(Counter(e.change.value for e in events)),
                "association_count": len(matches),
                "verified_identity_count": sum(m["identity_verified"] for m in matches),
            }
        )
    original = dict(
        Counter(e["change"] for e in report["events"] if e.get("object_id"))
    )
    if rows[0]["class_counts"] != original:
        raise ValueError(
            "Baseline observation replay does not reproduce original semantic classes"
        )
    result = {
        "inference_report_sha256": sha256(args.run / "report.json"),
        "mode": "Post-inference replay of saved actual neural observations and measured geometry; no model invocation in this ablation",
        "rows": rows,
        "calibrated_uncertainty_ablation": "Unavailable: no independent likelihood calibration or covariance estimator exists for this pair",
        "interpretation": "Identical abstaining outputs cannot establish benefit from any component; this is a feasibility ablation, not a general benchmark",
    }
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
