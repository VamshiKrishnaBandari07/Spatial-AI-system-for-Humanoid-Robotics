"""Native movement versus measured diagnostics; no covariance from ICP RMSE."""

import argparse
import json
from pathlib import Path
import numpy as np
from spatialmemory.native_review import review_native, read_ascii_instances
from spatialmemory.benchmarking import sha256


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--plan", type=Path, required=True)
    p.add_argument("--run", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    plan = json.loads(args.plan.read_text())
    review = review_native(plan)
    report = json.loads((args.run / "report.json").read_text())
    rows = []
    for obj in review["objects"]:
        if obj["change_type"] != "MOVED":
            continue
        shape = {}
        for state, scan, key in [
            ("before", plan["reference"], "native_before_id"),
            ("after", plan["rescan"], "native_after_id"),
        ]:
            points, ids = read_ascii_instances(
                Path(plan["raw_destination"])
                / scan
                / "labels.instances.annotated.v2.ply"
            )
            selected = points[ids == obj[key]]
            sides = np.ptp(selected, axis=0)
            volume = float(np.prod(sides))
            shape[state] = {
                "native_annotated_vertex_count": len(selected),
                "axis_extent_metres": sides.tolist(),
                "vertices_per_bbox_m3": len(selected) / volume if volume > 0 else None,
                "density_is_not_sensor_noise": True,
            }
        rows.append(
            {
                "object_id": obj["object_id"],
                "label": obj["label"],
                "source_audited_displacement_metres": obj[
                    "native_centroid_translation_metres"
                ],
                "native_shape_diagnostics": shape,
                "prediction_link_verified": False,
                "object_centroid_uncertainty": None,
                "depth_error_bound": None,
                "known_raw_depth_quantization_metres": 0.001,
                "quantization_is_not_sensor_accuracy": True,
                "viewpoint_coverage_for_correct_object": None,
                "reason": "No independently verified prediction-to-native object identity; unrelated detections cannot supply object uncertainty",
            }
        )
    result = {
        "source_report_sha256": sha256(args.run / "report.json"),
        "objects": rows,
        "registration_rmse_metres": report["registration"]["rmse"],
        "heldout_rmse_metres": report["registration"]["heldout_rmse"],
        "registration_covariance": None,
        "legacy_displacement_threshold": report["config"]["voxel_size"] * 2,
        "root_cause_scope": "Identity gate fails before movement classification. Coarse displacement threshold is an additional sensitivity limitation, not the executed primary cause.",
        "bounded_rule": "Displacement interval radius = two declared depth error budgets + declared alignment error budget + observed before/after repeat-view centroid spreads. Bounds must be supplied with provenance; they are not automatically calibrated or verified.",
        "missing_error_terms_keep_UNKNOWN": True,
        "calibration_split_used": None,
    }
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(
        json.dumps(
            {
                "supported_moved_objects": len(rows),
                "calibrated_uncertainty_available": False,
            }
        )
    )


if __name__ == "__main__":
    main()
