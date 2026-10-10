"""Validate real pair provenance and evaluate an existing completed inference report."""

import argparse
import json
import math
from pathlib import Path
from spatialmemory.inputs import Annotations
from spatialmemory.benchmarking import (
    PairEvidence,
    IdentityMapping,
    validate_pair,
    benchmark,
    sha256,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--pair-evidence", type=Path, required=True)
    parser.add_argument("--annotations", type=Path)
    parser.add_argument("--report", type=Path)
    parser.add_argument("--identity-mapping", type=Path)
    parser.add_argument("--resource-profile", type=Path)
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    try:
        evidence = PairEvidence.model_validate_json(args.pair_evidence.read_text())
        annotations = (
            Annotations.model_validate_json(args.annotations.read_text())
            if args.annotations
            else None
        )
        report = json.loads(args.report.read_text()) if args.report else None
        pair = validate_pair(
            args.manifest, evidence, args.project_root, annotations, report
        )
        (args.output / ("pair_validation_" + pair["scene_id"] + ".json")).write_text(
            json.dumps(pair, indent=2) + "\n"
        )
        if not args.identity_mapping:
            print(json.dumps(pair, indent=2))
            if pair["pair_validity_status"] != "VERIFIED_FOR_EVALUATION":
                (args.output / "benchmark_failure.json").write_text(
                    json.dumps(
                        {
                            "status": "blocked",
                            "error": "Pair validation incomplete: "
                            + "; ".join(pair["warnings"]),
                            "pair_status": pair["pair_validity_status"],
                            "metrics_computed": False,
                        },
                        indent=2,
                    )
                    + "\n"
                )
                parser.exit(
                    2, "Pair validation incomplete; supervised benchmark not run.\n"
                )
            return
        if annotations is None or report is None:
            raise ValueError("Benchmark requires annotations and a completed report")
        mapping = IdentityMapping.model_validate_json(args.identity_mapping.read_text())
        metrics = benchmark(report, annotations, pair, mapping)
        if args.resource_profile:
            profile = json.loads(args.resource_profile.read_text())
            if profile.get("status") != "passed" or profile.get(
                "report_sha256"
            ) != sha256(args.report):
                raise ValueError(
                    "Resource profile does not match the completed inference report"
                )
            memory = profile.get("peak_child_process_rss_mib")
            if (
                isinstance(memory, bool)
                or not isinstance(memory, (int, float))
                or not math.isfinite(memory)
                or memory <= 0
            ):
                raise ValueError("Invalid peak-memory measurement")
            metrics["peak_memory_mib"] = memory
            metrics["memory_measurement_method"] = profile["memory_method"]
            metrics["process_wall_seconds"] = profile["wall_seconds"]
        metrics["input_provenance"] = {
            str(p): sha256(p)
            for p in [
                args.manifest,
                args.pair_evidence,
                args.annotations,
                args.report,
                args.identity_mapping,
            ]
        }
        with (args.output / "benchmark_metrics.json").open("x") as destination:
            destination.write(json.dumps(metrics, indent=2) + "\n")
        print(json.dumps(metrics, indent=2))
    except (ValueError, OSError) as error:
        (args.output / "benchmark_failure.json").write_text(
            json.dumps(
                {"status": "blocked", "error": str(error), "metrics_computed": False},
                indent=2,
            )
            + "\n"
        )
        parser.exit(2, f"Benchmark blocked: {error}\n")


if __name__ == "__main__":
    main()
