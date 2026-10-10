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
        "3rscan": "No scan assets; official access approval required",
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
        f"The proposed ARKit recording combination is **{pair['pair_validity_status']}**: distinct visit IDs are not same-room correspondence evidence, physical changes are undocumented, shared cross-visit registration is unverified and independent change ground truth is absent. Heads has one sequence; Replica has no recordings; 3RScan has no downloaded scans. Thus there are **zero verified real annotated change pairs**.",
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
        "The ten real-data completion criteria are **not satisfied**. No genuine changed pair means full cross-visit registration, supervised end-to-end metrics and ablations cannot be executed. Source-based human annotation and approved dataset access remain prerequisites. Verified loader checks and one real-recording control do not resolve the benchmark data problem.",
        "",
        "## Workspace search and access preparation",
        "",
        "The complete workspace search found 44 videos and 15 annotation candidates, with no approved 3RScan scan assets or independently reviewed real change labels. Exact paths and video classifications are in workspace_data_search.json. Synthetic upload origins were checked against fixture SHA-256 hashes. Expected 3RScan and smartphone directories are empty placeholders, not acquired recordings. The empty smartphone annotation template and official access steps are documented in ../docs/DATASET_ACCESS.md.",
        "",
        "TUM RGB-D official documentation is accessible and confirms the default data license is CC BY 4.0. Its archive redirects to webshare.cvg.cit.tum.de, still blocked by runtime policy (HTTP tunnel 403). No TUM files are available. The additional archive host is saved in the environment draft and requires settings publication before retrying; see tum_access_check.json.",
    ]
    (folder / "dataset_quality_report.md").write_text("\n".join(rows) + "\n")
    status = {
        "completed": False,
        "verified_real_changed_pairs": 0,
        "verified_real_annotation_sets": 0,
        "real_benchmark_metrics_computed": False,
        "ablation_studies_run": False,
        "core_tests_passed": int(count.group(1)),
        "native_frames_checked": sum(s["frames_checked"] for s in native["sequences"]),
        "real_recording_control_completed": control["status"] == "complete",
        "completion_conditions": {
            "genuine_same_scene_changed_pair": False,
            "documented_physical_changes_for_pair": False,
            "both_pair_visits_reconstruction_usable": False,
            "shared_cross_visit_registration_verified": False,
            "ground_truth_provenance_verified": False,
            "full_pipeline_on_valid_pair": False,
            "detected_real_changes_visualized": False,
            "quantitative_real_ground_truth_results": False,
            "failures_and_limits_documented": True,
            "real_benchmark_reproducible": False,
        },
    }
    (folder / "completion_status.json").write_text(json.dumps(status, indent=2) + "\n")
    print(json.dumps(status, indent=2))


if __name__ == "__main__":
    main()
