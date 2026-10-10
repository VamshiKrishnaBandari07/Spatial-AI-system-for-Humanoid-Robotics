# Dataset quality report

Date: 2026-10-10. Generated from saved executed evidence; no benchmark labels or results inferred.

## Verified inventory

| Dataset | Storage MiB | Local capability |
| --- | ---: | --- |
| arkitscenes | 1101.9 | Two source recordings; RGB, native depth, variable per-frame K, timestamps and camera-to-world poses |
| seven_scenes | 964.9 | Heads seq-02; RGB, depth, poses and nominal depth K; RGB calibration/extrinsics unavailable |
| replica | 4276.8 | One finite 5616×5616×3 float16 Ptex atlas and incomplete release prefix; no geometry/camera recording |
| 3rscan | 0.0 | No scan assets; official access approval required |

## Executed input validation

- ARKitScenes 47333462: 48 frames checked; status passed; variable intrinsics True; maximum pinhole projection round trip 5.68e-14 pixels.
- ARKitScenes 41069050: 48 frames checked; status passed; variable intrinsics True; maximum pinhole projection round trip 8.53e-14 pixels.
- 7-Scenes seq-02: 48 frames checked; status passed; variable intrinsics False; maximum pinhole projection round trip 1.21e-13 pixels.

Actual source-depth conversion, invalid-value handling, per-frame K, pose inversion, camera/world transforms, source timestamps and synchronization checks passed. Native ARKit depth clouds were generated under ignored `data/datasets/recovery_validation`. Cross-view native sensor-depth residuals are diagnostics, not model accuracy. Unregistered 7-Scenes RGB/depth is excluded from aligned fusion. Replica is appearance-only.

Archive verification: 7 retained ZIP archives, 19,531 payload members read with CRC checks. Local hashes match acquisition journals; independent upstream checksums are not claimed. Windows thumbnail caches are excluded without deletion.

## Pair and annotation readiness

The proposed ARKit recording combination is **REJECTED**: distinct visit IDs are not same-room correspondence evidence, physical changes are undocumented, shared cross-visit registration is unverified and independent change ground truth is absent. Heads has one sequence; Replica has no recordings; 3RScan has no downloaded scans. Thus there are **zero verified real annotated change pairs**.

The annotation workspace supports independent playback, persistent object IDs, before/after observation association, UNKNOWN, import/review and validated export. This task created no real human annotations. Browser fixtures are explicitly synthetic. Reviews are page-local and declarations are not authenticated. Richer annotation levels are supported by the typed schema; geometry asset references do not constitute validated IoU ground truth.

## Actual real-recording inference control

Existing pretrained consensus pipeline ran on ARKitScenes recording 47333462 supplied twice, as a loader/inference control. Runtime: 171.7 seconds; complete child-process wall time: 173.1 seconds; peak individual child RSS: 9961.1 MiB. Output classes: `{'UNKNOWN': 1}`. Estimated semantic labels are unverified. This is RGB-only reconstruction; optional depth/poses are retained, not fused.

Shared identity alignment and finite arrays passed, but identical-input registration residuals are not independent geometric accuracy. The model produced UNKNOWN with only one distinct object view per visit; no rule threshold was relaxed to force a decision. This is not a genuine two-visit benchmark. Precision, recall, F1, false-removal rates, localization, calibration and ablation results remain unavailable.

## Tests and reproduction

Independent clean CPU suite: **92 passed**. The complete output is `regression_test_output.txt`, including one dependency deprecation warning. Development suite also passed. Annotation browser check: **passed**; evidence in `annotation_browser_check.json`. Existing edge-case UI replay also passed using the stored synthetic report.

Executed commands (from the repository, with the installed environment activated):

```bash
python scripts/audit_datasets.py --data-root data/datasets --output reports
python scripts/validate_reconstruction_inputs.py \
  --sequence data/datasets/arkitscenes/processed/47333462_synchronized/sequence.json \
  --sequence data/datasets/arkitscenes/processed/41069050/sequence.json \
  --sequence data/datasets/seven_scenes/processed/heads_seq02/sequence.json \
  --replica-atlas data/datasets/replica/raw/apartment_0/textures/27-color-ptex.hdr \
  --assets data/datasets/recovery_validation --output reports/reconstruction_input_validation.json
python -m pytest -q
SPATIALMEMORY_URL=http://127.0.0.1:8001 node scripts/check_annotation_workspace.cjs
python scripts/profile_inference.py \
  --manifest data/datasets/recovery_controls/manifest.json \
  --config configs/consensus_cpu.json --project-root . \
  --output /workspace/spatialmemory-real-recording-control-20261010
python scripts/build_dataset_quality_report.py --reports reports
```

Profile output is immutable: select a new output directory for another inference run. Native assets/manifests are environment-specific and are not bundled in Git; obtain them through the official sources and preparation commands in `docs/DATASETS.md`. The negative ARKit pair-validation command exited 2 intentionally, with no supervised metrics. See `pair_validation_arkit_unverified_pair.json` and `benchmark_failure.json`.

## Licensing, coverage and remaining blockers

Apple ARKitScenes source licence and Microsoft MSR-LA are retained with raw data; 7-Scenes is noncommercial research. Replica terms restrict research/educational use and no complete geometry asset is claimed. VGGT's checkpoint is CC-BY-NC-4.0. Code licences do not grant dataset rights. No raw images, videos, point clouds or checkpoints are published.

3RScan/RIO10 require official access approval. ETH Zurich and TU Wien download/source hosts are denied by current runtime policy; additive domain requirements are saved but not active. Their dataset licences, pair sizes and annotation files remain unverified. No forms/emails were submitted. No full archives were downloaded to work around access restrictions. See `dataset_selection.md` and `dataset_source_checks.json`.

The ten real-data completion criteria are **not satisfied**. No genuine changed pair means full cross-visit registration, supervised end-to-end metrics and ablations cannot be executed. Source-based human annotation and approved dataset access remain prerequisites. Verified loader checks and one real-recording control do not resolve the benchmark data problem.

## Workspace search and access preparation

The complete workspace search found 44 videos and 15 annotation candidates, with no approved 3RScan scan assets or independently reviewed real change labels. Exact paths and video classifications are in workspace_data_search.json. Synthetic upload origins were checked against fixture SHA-256 hashes. Expected 3RScan and smartphone directories are empty placeholders, not acquired recordings. The empty smartphone annotation template and official access steps are documented in ../docs/DATASET_ACCESS.md.

TUM RGB-D official page/archive requests were blocked by current network policy (HTTP tunnel 403). No TUM data or calibration is claimed available. The cvg.cit.tum.de addition is saved in the environment draft and requires settings publication before retrying; see tum_access_check.json.
