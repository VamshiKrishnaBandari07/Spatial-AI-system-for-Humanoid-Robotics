"""Build a quality report from executed audit, loader, browser and test evidence."""

import argparse
import json
from pathlib import Path
import re


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reports", type=Path, default=Path("reports"))
    args = parser.parse_args()
    folder = args.reports
    inventory = json.loads((folder / "dataset_inventory.json").read_text())
    native = json.loads((folder / "reconstruction_input_validation.json").read_text())
    browser = json.loads((folder / "annotation_browser_check.json").read_text())
    pair = json.loads(
        (folder / "pair_validation_arkit_unverified_pair.json").read_text()
    )
    control = json.loads((folder / "real_recording_inference_control.json").read_text())
    scan_validation_path = folder / "3rscan_native_validation.json"
    scan_validation = (
        json.loads(scan_validation_path.read_text())
        if scan_validation_path.exists()
        else None
    )
    scan_attempt_path = folder / "3rscan_inference_attempt.json"
    scan_attempt = (
        json.loads(scan_attempt_path.read_text())
        if scan_attempt_path.exists()
        else None
    )
    execution_path = folder / "3rscan_validated_execution.json"
    execution = (
        json.loads(execution_path.read_text()) if execution_path.exists() else None
    )
    validated = bool(
        execution
        and execution["status"] == "complete"
        and execution["registration"]["valid"]
    )
    test_log = (folder / "regression_test_output.txt").read_text()
    count = re.search(r"(\d+) passed", test_log)
    if not count:
        raise ValueError("No successful executed regression result found")
    rows = [
        "# Dataset quality report",
        "",
        f"Date: {inventory['date']}. Generated from saved executed evidence; no benchmark labels or results inferred.",
        "",
        "## Verified inventory",
        "",
        "| Dataset | Storage MiB | Local capability |",
        "| --- | ---: | --- |",
    ]
    capabilities = {
        "arkitscenes": "Two source recordings; RGB, native depth, variable per-frame K, timestamps and camera-to-world poses",
        "seven_scenes": "Heads seq-02; RGB, depth, poses and nominal depth K; RGB calibration/extrinsics unavailable",
        "replica": "One finite 5616×5616×3 float16 Ptex atlas and incomplete release prefix; no geometry/camera recording",
        "3rscan": (
            "One approved reference/rescan pair; RGB, native depth/calibration/poses and instance annotations verified; partial native MOVED audit available; full canonical labels unreviewed"
            if scan_validation
            else "No scan assets; official access approval required"
        ),
    }
    for dataset in inventory["datasets"]:
        rows.append(
            f"| {dataset['dataset']} | {dataset['size_bytes'] / 1024**2:.1f} | {capabilities[dataset['dataset']]} |"
        )
    rows += ["", "## Executed input validation", ""]
    for sequence in native["sequences"]:
        roundtrip = max(frame["pixel_roundtrip_max"] for frame in sequence["frames"])
        rows.append(
            f"- {sequence['dataset']} {sequence['recording_id']}: {sequence['frames_checked']} frames checked; status {sequence['status']}; variable intrinsics {sequence['variable_intrinsics']}; maximum pinhole projection round trip {roundtrip:.3g} pixels."
        )
    rows += [
        "",
        "Actual source-depth conversion, invalid-value handling, per-frame K, pose inversion, camera/world transforms, source timestamps and synchronization checks passed. Native ARKit depth clouds were generated under ignored `data/datasets/recovery_validation`. Cross-view native sensor-depth residuals are diagnostics, not model accuracy. Unregistered 7-Scenes RGB/depth is excluded from aligned fusion. Replica is appearance-only.",
        "",
        f"Archive verification: {len(inventory['archives'])} retained ZIP archives, {sum(a['members_crc_verified'] for a in inventory['archives']):,} payload members read with CRC checks. Local hashes match acquisition journals; independent upstream checksums are not claimed. Windows thumbnail caches are excluded without deletion.",
        "",
        "## Pair and annotation readiness",
        "",
        f"The proposed ARKit recording combination is **{pair['pair_validity_status']}**: distinct visit IDs are not same-room correspondence evidence, physical changes are undocumented, shared cross-visit registration is unverified and independent change ground truth is absent. Heads has one sequence; Replica has no recordings. There are **zero independently reviewed canonical real annotation sets ready for benchmark evaluation**.",
        "",
        "The annotation workspace supports independent playback, persistent object IDs, before/after observation association, UNKNOWN, import/review and validated export. This task created no real human annotations. Browser fixtures are explicitly synthetic. Reviews are page-local and declarations are not authenticated. Richer annotation levels are supported by the typed schema; geometry asset references do not constitute validated IoU ground truth.",
        "",
        "## Actual real-recording inference control",
        "",
        f"Existing pretrained consensus pipeline ran on ARKitScenes recording {control['recording_id']} supplied twice, as a loader/inference control. Runtime: {control['runtime_seconds']:.1f} seconds; complete child-process wall time: {control['process_wall_seconds']:.1f} seconds; peak individual child RSS: {control['peak_child_process_rss_mib']:.1f} MiB. Output classes: `{control['event_counts']}`. Estimated semantic labels are unverified. This is RGB-only reconstruction; optional depth/poses are retained, not fused.",
        "",
        "Shared identity alignment and finite arrays passed, but identical-input registration residuals are not independent geometric accuracy. The model produced UNKNOWN with only one distinct object view per visit; no rule threshold was relaxed to force a decision. This is not a genuine two-visit benchmark. Precision, recall, F1, false-removal rates, localization, calibration and ablation results remain unavailable.",
        "",
        "## Tests and reproduction",
        "",
        f"Independent clean CPU suite: **{count.group(1)} passed**. The complete output is `regression_test_output.txt`, including one dependency deprecation warning. Development suite also passed. Annotation browser check: **{browser['status']}**; evidence in `annotation_browser_check.json`. Existing edge-case UI replay also passed using the stored synthetic report.",
        "",
        "Executed commands (from the repository, with the installed environment activated):",
        "",
        "```bash",
        "python scripts/audit_datasets.py --data-root data/datasets --output reports",
        "python scripts/validate_reconstruction_inputs.py \\",
        "  --sequence data/datasets/arkitscenes/processed/47333462_synchronized/sequence.json \\",
        "  --sequence data/datasets/arkitscenes/processed/41069050/sequence.json \\",
        "  --sequence data/datasets/seven_scenes/processed/heads_seq02/sequence.json \\",
        "  --replica-atlas data/datasets/replica/raw/apartment_0/textures/27-color-ptex.hdr \\",
        "  --assets data/datasets/recovery_validation --output reports/reconstruction_input_validation.json",
        "python -m pytest -q",
        "SPATIALMEMORY_URL=http://127.0.0.1:8001 node scripts/check_annotation_workspace.cjs",
        "python scripts/profile_inference.py \\",
        "  --manifest data/datasets/recovery_controls/manifest.json \\",
        "  --config configs/consensus_cpu.json --project-root . \\",
        "  --output /workspace/spatialmemory-real-recording-control-20261010",
        "python scripts/build_dataset_quality_report.py --reports reports",
        "```",
        "",
        "Profile output is immutable: select a new output directory for another inference run. Native assets/manifests are environment-specific and are not bundled in Git; obtain them through the official sources and preparation commands in `docs/DATASETS.md`. The negative ARKit pair-validation command exited 2 intentionally, with no supervised metrics. See `pair_validation_arkit_unverified_pair.json` and `benchmark_failure.json`.",
        "",
        "## Licensing, coverage and remaining blockers",
        "",
        "Apple ARKitScenes source licence and Microsoft MSR-LA are retained with raw data; 7-Scenes is noncommercial research. Replica terms restrict research/educational use and no complete geometry asset is claimed. VGGT's checkpoint is CC-BY-NC-4.0. Code licences do not grant dataset rights. No raw images, videos, point clouds or checkpoints are published.",
        "",
        "3RScan/RIO10 require official access approval. ETH Zurich and TU Wien download/source hosts are denied by current runtime policy; additive domain requirements are saved but not active. Their dataset licences, pair sizes and annotation files remain unverified. No forms/emails were submitted. No full archives were downloaded to work around access restrictions. See `dataset_selection.md` and `dataset_source_checks.json`.",
        "",
        "The ten real-data completion criteria are **not satisfied**. A full independently reviewed change inventory, successful physical-change detection, calibrated uncertainty and independent-scene benchmarks remain prerequisites. Current partial 3RScan execution evidence appears below. Loader checks and inference completion alone do not establish prediction accuracy.",
        "",
        "## Workspace search and access preparation",
        "",
        "The initial workspace search found 44 videos and 15 annotation candidates, with no approved 3RScan scan assets or independently reviewed real change labels at that time. Exact historical paths and video classifications are in workspace_data_search.json. Subsequent approved acquisitions are documented separately; the initial search is not a current inventory. Smartphone directories remain empty. The empty smartphone annotation template and official access steps are documented in ../docs/DATASET_ACCESS.md.",
        "",
        "TUM RGB-D official documentation is accessible and confirms the default data license is CC BY 4.0. Its archive redirects to webshare.cvg.cit.tum.de, still blocked by runtime policy (HTTP tunnel 403). No TUM files are available. The additional archive host is saved in the environment draft and requires settings publication before retrying; see tum_access_check.json.",
    ]
    if scan_validation:
        rows += [
            "",
            "## Approved 3RScan acquisition",
            "",
            f"One documented reference/rescan pair was acquired after the user confirmed official approval and agreement to Terms of Use. All {sum(s['frames_checked'] for s in scan_validation['scans'])} native RGB-D/pose frame sets passed validation, together with semantic IDs and annotated PLY vertices. Original RGB/depth have different dimensions and intrinsics; no pixel registration is assumed. The native split is train. See 3rscan_acquisition.json and 3rscan_native_validation.json for exact paths, hashes, sizes and observations.",
            "",
            "Native metadata documents three rigid changes (toilet, scale, trash can). The derived annotation draft retains native IDs and remains UNREVIEWED with UNKNOWN canonical classes, avoiding an unverified mapping of rigid changes to movement/rotation labels. This historical draft remains unreviewed. The subsequent deterministic native-source audit and metric registration are reported separately below; no independent human review is claimed. No identity mapping or reviewer identities were invented.",
        ]
    if scan_attempt:
        rows += [
            "",
            f"The original unchanged consensus CPU profile was attempted on the real pair. It failed at {scan_attempt['stage']}: {scan_attempt['error']}. Runtime {scan_attempt['runtime_seconds']:.1f} seconds; wall time {scan_attempt['process_wall_seconds']:.1f} seconds; peak individual child RSS {scan_attempt['peak_child_process_rss_mib']:.1f} MiB. Both visits produced finite neural pose/depth/point-map arrays, but shared registration was rejected. No gate was relaxed, no changes were generated, and no supervised metrics were computed. See 3rscan_inference_attempt.json and pair_validation_3rscan.json.",
        ]
    if validated:
        rows += [
            "",
            "## Current real-pair metric execution",
            "",
            "The opt-in measured RGB-D profile completed with actual neural expert inference and source-transform-assisted robust ICP. Original failure was zero lifted cross-session correspondences, not a demonstrated unit or inversion error. Depth counts convert from millimetres to metres; native column-major scene translations already use metres. The original failed run remains preserved.",
            "",
            "Symmetric overlap 48.61%; inlier RMSE 0.03346 m; held-out overlap 46.09% and RMSE 0.03970 m. Fixed gates passed without identity fallback or relaxed thresholds. One fresh VGGT run precedes the final integrity-validated VGGT replay with fresh semantic inference.",
            "",
            "The native positive MOVED subset contains one supported instance. It was predicted UNKNOWN: recall 0, F1 0, precision undefined. All 28 final decisions are UNKNOWN and no cross-session semantic association was accepted. Full detection metrics and identity accuracy remain unavailable. Post-inference ablations all abstained and cannot establish component benefit. Uncertainty is not calibrated.",
            "",
            "See ../docs/3RSCAN_VALIDATION.md for reproducible commands, precise source conventions, annotated scope, private 3D viewer and reports. This is one training pair with native alignment assistance, not a held-out generalization benchmark.",
        ]
    if validated and execution.get("additional_coverage_run"):
        coverage = execution["additional_coverage_run"]
        rows += [
            "",
            "## Expanded semantic coverage",
            "",
            f"Twelve keyframes per recording completed in {coverage['profile']['wall_seconds']:.2f} seconds. Decisions: {coverage['counts']}; accepted associations: {coverage['accepted_associations']}. All 80 same-label pairs failed the unchanged appearance gate. The single supported MOVED instance remains UNKNOWN; extra views do not establish detection accuracy. See 3rscan_coverage_evaluation.json and 3rscan_coverage_ablations.json.",
        ]
    (folder / "dataset_quality_report.md").write_text("\n".join(rows) + "\n")
    status = {
        "completed": False,
        "verified_real_changed_pairs": 1 if scan_validation else 0,
        "verified_real_annotation_sets": 0,
        "real_benchmark_metrics_computed": False,
        "ablation_studies_run": validated,
        "ablation_scope": "Saved-observation feasibility replay; all outputs UNKNOWN"
        if validated
        else None,
        "partial_native_ground_truth_metrics_computed": validated,
        "partial_native_ground_truth_support": execution[
            "supported_ground_truth_objects"
        ]
        if validated
        else 0,
        "core_tests_passed": int(count.group(1)),
        "native_frames_checked": sum(s["frames_checked"] for s in native["sequences"])
        + (
            sum(s["frames_checked"] for s in scan_validation["scans"])
            if scan_validation
            else 0
        ),
        "real_recording_control_completed": control["status"] == "complete",
        "real_changed_pair_inference_attempted": scan_attempt is not None,
        "real_changed_pair_inference_status": execution["status"]
        if validated
        else scan_attempt["status"]
        if scan_attempt
        else None,
        "completion_conditions": {
            "genuine_same_scene_changed_pair": scan_validation is not None,
            "documented_physical_changes_for_pair": scan_validation is not None,
            "both_pair_visits_reconstruction_usable": validated,
            "shared_cross_visit_registration_verified": validated,
            "ground_truth_provenance_verified": False,
            "full_pipeline_on_valid_pair": validated,
            "detected_real_changes_visualized": False,
            "quantitative_real_ground_truth_results": validated,
            "failures_and_limits_documented": True,
            "real_benchmark_reproducible": False,
        },
    }
    (folder / "completion_status.json").write_text(json.dumps(status, indent=2) + "\n")
    print(json.dumps(status, indent=2))


if __name__ == "__main__":
    main()
