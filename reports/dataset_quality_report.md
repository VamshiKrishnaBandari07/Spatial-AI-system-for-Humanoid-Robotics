# Dataset quality report

Date: 2026-10-10. Generated from saved executed evidence; no benchmark labels or results inferred.

## Verified inventory

| Dataset | Storage MiB | Local capability |
| --- | ---: | --- |
| arkitscenes | 1101.9 | Two source recordings; RGB, native depth, variable per-frame K, timestamps and camera-to-world poses |
| seven_scenes | 964.9 | Heads seq-02; RGB, depth, poses and nominal depth K; RGB calibration/extrinsics unavailable |
| replica | 4276.8 | One finite 5616×5616×3 float16 Ptex atlas and incomplete release prefix; no geometry/camera recording |
| 3rscan | 168.6 | One approved reference/rescan pair; RGB, native depth/calibration/poses and instance annotations verified; partial native MOVED audit available; full canonical labels unreviewed |

## Executed input validation

- ARKitScenes 47333462: 48 frames checked; status passed; variable intrinsics True; maximum pinhole projection round trip 5.68e-14 pixels.
- ARKitScenes 41069050: 48 frames checked; status passed; variable intrinsics True; maximum pinhole projection round trip 8.53e-14 pixels.
- 7-Scenes seq-02: 48 frames checked; status passed; variable intrinsics False; maximum pinhole projection round trip 1.21e-13 pixels.

Actual source-depth conversion, invalid-value handling, per-frame K, pose inversion, camera/world transforms, source timestamps and synchronization checks passed. Native ARKit depth clouds were generated under ignored `data/datasets/recovery_validation`. Cross-view native sensor-depth residuals are diagnostics, not model accuracy. Unregistered 7-Scenes RGB/depth is excluded from aligned fusion. Replica is appearance-only.

Archive verification: 9 retained ZIP archives, 20,304 payload members read with CRC checks. Local hashes match acquisition journals; independent upstream checksums are not claimed. Windows thumbnail caches are excluded without deletion.

## Pair and annotation readiness

The proposed ARKit recording combination is **REJECTED**: distinct visit IDs are not same-room correspondence evidence, physical changes are undocumented, shared cross-visit registration is unverified and independent change ground truth is absent. Heads has one sequence; Replica has no recordings. There are **zero independently reviewed canonical real annotation sets ready for benchmark evaluation**.

The annotation workspace supports independent playback, persistent object IDs, before/after observation association, UNKNOWN, import/review and validated export. This task created no real human annotations. Browser fixtures are explicitly synthetic. Reviews are page-local and declarations are not authenticated. Richer annotation levels are supported by the typed schema; geometry asset references do not constitute validated IoU ground truth.

## Actual real-recording inference control

Existing pretrained consensus pipeline ran on ARKitScenes recording 47333462 supplied twice, as a loader/inference control. Runtime: 171.7 seconds; complete child-process wall time: 173.1 seconds; peak individual child RSS: 9961.1 MiB. Output classes: `{'UNKNOWN': 1}`. Estimated semantic labels are unverified. This is RGB-only reconstruction; optional depth/poses are retained, not fused.

Shared identity alignment and finite arrays passed, but identical-input registration residuals are not independent geometric accuracy. The model produced UNKNOWN with only one distinct object view per visit; no rule threshold was relaxed to force a decision. This is not a genuine two-visit benchmark. Precision, recall, F1, false-removal rates, localization, calibration and ablation results remain unavailable.

## Tests and reproduction

Independent clean CPU suite: **129 passed**. The current output is `application_tests.txt`, including one dependency deprecation warning. The latest full-suite evidence is from the clean CPU environment. Annotation browser check: **passed**; evidence in `annotation_browser_check.json`. Existing edge-case UI replay also passed using the stored synthetic report.

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

The ten real-data completion criteria are **not satisfied**. A full independently reviewed change inventory, successful physical-change detection, calibrated uncertainty and independent-scene benchmarks remain prerequisites. Current partial 3RScan execution evidence appears below. Loader checks and inference completion alone do not establish prediction accuracy.

## Workspace search and access preparation

The initial workspace search found 44 videos and 15 annotation candidates, with no approved 3RScan scan assets or independently reviewed real change labels at that time. Exact historical paths and video classifications are in workspace_data_search.json. Subsequent approved acquisitions are documented separately; the initial search is not a current inventory. Smartphone directories remain empty. The empty smartphone annotation template and official access steps are documented in ../docs/DATASET_ACCESS.md.

TUM RGB-D official documentation is accessible and confirms the default data license is CC BY 4.0. Its archive redirects to webshare.cvg.cit.tum.de, still blocked by runtime policy (HTTP tunnel 403). No TUM files are available. The additional archive host is saved in the environment draft and requires settings publication before retrying; see tum_access_check.json.

## Approved 3RScan acquisition

One documented reference/rescan pair was acquired after the user confirmed official approval and agreement to Terms of Use. All 257 native RGB-D/pose frame sets passed validation, together with semantic IDs and annotated PLY vertices. Original RGB/depth have different dimensions and intrinsics; no pixel registration is assumed. The native split is train. See 3rscan_acquisition.json and 3rscan_native_validation.json for exact paths, hashes, sizes and observations.

Native metadata documents three rigid changes (toilet, scale, trash can). The derived annotation draft retains native IDs and remains UNREVIEWED with UNKNOWN canonical classes, avoiding an unverified mapping of rigid changes to movement/rotation labels. This historical draft remains unreviewed. The subsequent deterministic native-source audit and metric registration are reported separately below; no independent human review is claimed. No identity mapping or reviewer identities were invented.

The original unchanged consensus CPU profile was attempted on the real pair. It failed at shared_registration: Registration gate failed: Insufficient nondegenerate consensus. Runtime 346.2 seconds; wall time 348.4 seconds; peak individual child RSS 9966.8 MiB. Both visits produced finite neural pose/depth/point-map arrays, but shared registration was rejected. No gate was relaxed, no changes were generated, and no supervised metrics were computed. See 3rscan_inference_attempt.json and pair_validation_3rscan.json.

## Current real-pair metric execution

The opt-in measured RGB-D profile completed with actual neural expert inference and source-transform-assisted robust ICP. Original failure was zero lifted cross-session correspondences, not a demonstrated unit or inversion error. Depth counts convert from millimetres to metres; native column-major scene translations already use metres. The original failed run remains preserved.

Symmetric overlap 48.61%; inlier RMSE 0.03346 m; held-out overlap 46.09% and RMSE 0.03970 m. Fixed gates passed without identity fallback or relaxed thresholds. One fresh VGGT run precedes the final integrity-validated VGGT replay with fresh semantic inference.

The native positive MOVED subset contains one supported instance. It was predicted UNKNOWN: recall 0, F1 0, precision undefined. All 28 final decisions are UNKNOWN and no cross-session semantic association was accepted. Full detection metrics and identity accuracy remain unavailable. Post-inference ablations all abstained and cannot establish component benefit. Uncertainty is not calibrated.

See ../docs/3RSCAN_VALIDATION.md for reproducible commands, precise source conventions, annotated scope, private 3D viewer and reports. This is one training pair with native alignment assistance, not a held-out generalization benchmark.

## Expanded semantic coverage

Twelve keyframes per recording completed in 611.03 seconds. Decisions: {'UNKNOWN': 69}; accepted associations: 0. All 80 same-label pairs failed the unchanged appearance gate. The single supported MOVED instance remains UNKNOWN; extra views do not establish detection accuracy. See 3rscan_coverage_evaluation.json and 3rscan_coverage_ablations.json.

## Semantic recovery audit

All 69 saved UNKNOWN decisions are audited: 50 appearance-gate rejections, 8 no exact-label candidates, 11 geometric differences without positive free space. DINOv2 replay reproduces stored descriptors; independent matching labels are absent, so the appearance floor is unchanged. Configurable multi-cue assignment and declared-error-budget movement tests pass. The real repeat retains identical alignment, all 28 decisions UNKNOWN and missed native MOVED support one. The standard export now includes a self-contained scene viewer. See semantic_recovery_delivery.json and ../docs/HUMANOID_SUBMISSION.md; genuine phone execution remains unrun.

## Integrated application execution

A fresh neural UI upload of an identical Chess recording completed. Per-candidate gates, source visibility and a complete result ZIP were inspected. All changes remain UNKNOWN; this control does not supply real change-detection accuracy. See APPLICATION_VERIFICATION.md and application_verification.json. Genuine phone validation and independent identity/calibration data remain unavailable.
