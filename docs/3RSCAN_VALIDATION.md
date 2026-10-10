# Approved 3RScan validation — 2026-10-10

The actual approved reference/rescan pair now completes metric inference, reconstruction export and interactive evidence visualization. **The detection milestone remains incomplete:** all 28 decisions are UNKNOWN, including the one source-audited positive MOVED instance. This is one native training pair with source-transform-assisted registration, not a held-out benchmark or proof of generalization.

## Registration diagnosis and correction

Original error: `Insufficient nondegenerate consensus` at `shared_registration`. The exact saved neural arrays and original selected frames were inspected. Four cross-recording image comparisons produced mutual SIFT counts 1, 0, 0, 0. Confidence filtering removed the sole match, leaving **zero lifted 3D correspondences** and no RANSAC input. Original registration used neither native sensor depth nor known poses. No evidence established a unit or inversion error as its cause. See `reports/3rscan_registration_root_cause.json`; the returned identity-shaped invalid diagnostic was never accepted as an alignment.

The official native source was inspected at commit `12f040d3dc849e394ed366acd7b54e63acb49205`: `rio_lib/data.cc` reads Eigen matrices column-major; `rio.cc` applies the scene transform directly to metre-valued PLY points; `sequence.cc` applies a ×1000 conversion only when millimetre poses are requested. This resolves the misleading FAQ translation-unit wording: native JSON scene translations already use metres. Depth PGM counts use millimetres with depthShift=1000. Camera poses are `T_world_camera`; projecting world points uses its inverse. Camera coordinates are right/down/forward, with proper rigid rotations, not an inferred robot gravity frame.

RGB is 960×540 and depth 224×172, with different calibrated intrinsics. Depth-to-RGB projection uses supplied extrinsics and a nearest-depth z-buffer; uncovered pixels remain missing, never free-space evidence. Derived 30-fps frame times are explicitly conversion times, not unavailable native hardware timestamps. Original frame indices remain preserved.

Verified native `T_reference_rescan` initializes trimmed point-to-point ICP at 0.08/0.04/0.02 m voxel scales. Every fifth voxel sample is held out from refinement. Fixed gates require bidirectional overlap ≥0.35, RMSE ≤0.048 m at a 0.08 m distance threshold, correction ≤0.15 m and ≤10°, and a proper rigid transform. The sparse sensor trial failed held-out overlap (33.14%); denser coverage passed without lowering gates. This is source-assisted local refinement, not autonomous global registration or static-object-masked ICP. Robust trimming reduces outliers but does not guarantee all changed foreground is excluded.

Corrected transform, source rescan-world to destination reference-world, translations in metres:

```text
 0.8017031844 -0.5977159044 -0.0027952829  0.0946861297
 0.5977113834  0.8016512636  0.0098196912 -0.1070944779
-0.0036285432 -0.0095432498  0.9999479121  0.0166775517
 0             0             0             1
```

| Measurement | Result |
| --- | ---: |
| Source fitness | 69.63% |
| Bidirectional minimum overlap | 48.61% |
| Accepted inliers / source voxel points | 21,193 / 30,435 |
| Inlier RMSE | 0.033464 m |
| Held-out minimum overlap | 46.09% |
| Held-out inliers / source points | 4,035 / 6,087 |
| Held-out RMSE | 0.039702 m |
| ICP correction | 0.010712 m; 0.641727° |

Nearest-neighbor residuals measure fit on this pair, not registration covariance or independent scene accuracy. Full values are in `reports/3rscan_dense_alignment.json` and `reports/3rscan_partial_evaluation.json`.

## Ground truth and measured detection result

Reference `02b33dfb-be2b-2d54-92d2-cd012b2b3c40`; rescan `fcf66d9e-622d-291c-84c2-bb23dfe31327`. All 257 native RGB/depth/pose frame sets and 40,097 annotated vertices were checked. Source hashes and explicit native instance correspondences are retained. Missing instance IDs do not establish addition or removal.

Three explicit native rigid correspondences exist. Composing the scene transform with each native object transform gives canonical-centroid translations: toilet 0.01886 m, scale 0.00843 m and trash can 0.06921 m. With the declared 0.03 m translational review margin, only trash-can instance 27 supports MOVED. Sub-margin translation, rotation-only changes and all other classes remain UNKNOWN; no UNCHANGED labels are inferred from absent change entries. This is a deterministic source-integrity audit, **not independent human annotation review**. Full source-derived positions stay in ignored local `evaluation_review.json`; published review contains computed summaries only.

The final run contains 11 geometric UNKNOWN regions and 17 semantic UNKNOWN decisions, with zero accepted cross-session associations. Five before and twelve after semantic objects were formed. Accepted positive detection count is zero. The fixed trash/trash-can semantic alias and 0.25 m location gate yield no unambiguous prediction for the supported native positive:

| Supported subset | Precision | Recall | F1 | Support |
| --- | --- | ---: | ---: | ---: |
| MOVED | Undefined (no positive predictions) | 0 | 0 | 1 |

Confusion matrix on that subset: MOVED→MOVED 0; MOVED→UNKNOWN 1. Unsupported categories are excluded. Full inventory detection precision/recall/F1 and object matching accuracy are unavailable, not zero. Native alignment assistance and the preserved training split must accompany every use of these results.

Baseline, removing the multi-view gate, removing visibility surfaces and removing the observed spread/RMSE band all yield 17 semantic UNKNOWN decisions and zero associations. These saved-observation ablations cannot establish component benefit. A calibrated-uncertainty ablation is unavailable: raw detector/model confidence, observed position spread and registration RMSE do not implement calibrated posterior uncertainty.

Actual saved VGGT depth was also compared with projected measured depth after per-frame scale fitting on alternating pixels. Held-out median absolute errors for four frames are 0.00377, 0.01348, 0.02207 and 0.02528 m. Pixels are spatially correlated and scales were fitted; this is a limited diagnostic, not room-level reconstruction accuracy or metric monocular validation.

## Actual model invocation and execution

A fresh run at `/workspace/spatialmemory-3rscan-metric-final-20261010` invoked VGGT, Grounding DINO, SAM 2 and DINOv2. Wall time 227.29 s, peak individual child RSS 9,968.52 MiB. The final run at `/workspace/spatialmemory-3rscan-reviewed-final-20261010` uses quality-based semantic frame selection and integrity-validated saved VGGT predictions from that fresh run, invoking semantic experts again: wall time 169.03 s, inference time 163.94 s, peak individual child RSS 3,855.47 MiB. It must not be described as fresh VGGT inference.

Primary shared geometry is calibrated measured RGB-D; VGGT is actually executed for neural cross-validation, not substituted with a heuristic. The final report retains a legacy VGGT workflow string and generic scene-graph distance-unit label; this profile's numeric geometry and graph distances are metres. Neural depth scale remains separately fitted and unverified as a globally metric model. Exact input/report/config hashes, seed 7, dependency versions and profiling commands are in `reports/3rscan_validated_execution.json`. No training was started or restarted.

## Reproducible commands and private demo

Activate `/workspace/spatialmemory-venv` from the repository. Approved assets must already exist; they are not in Git. The explicit native source plan is `reports/3rscan_candidate_download_plan.json` and canonical manifest is:

```text
data/datasets/3rscan/processed/02b33dfb-be2b-2d54-92d2-cd012b2b3c40__fcf66d9e-622d-291c-84c2-bb23dfe31327/manifest.json
```

```bash
PAIR_MANIFEST=data/datasets/3rscan/processed/02b33dfb-be2b-2d54-92d2-cd012b2b3c40__fcf66d9e-622d-291c-84c2-bb23dfe31327/manifest.json
python scripts/prepare_metric_3rscan.py \
  --plan reports/3rscan_candidate_download_plan.json --manifest "$PAIR_MANIFEST"
# Choose a new output directory; profiling preserves existing results.
python scripts/profile_inference.py --manifest "$PAIR_MANIFEST" \
  --config configs/3rscan_metric_cpu.json --project-root . \
  --output /workspace/spatialmemory-3rscan-fresh-reproduction
# Existing final evidence run used configs/3rscan_metric_replay.json, whose reuse path
# explicitly points to the preserved fresh model run documented above.
MPLCONFIGDIR=/tmp/spatialmemory-matplotlib XDG_CACHE_HOME=/tmp/spatialmemory-font-cache \
python scripts/review_3rscan.py --plan reports/3rscan_candidate_download_plan.json \
  --run /workspace/spatialmemory-3rscan-reviewed-final-20261010 \
  --report-output reports/3rscan_partial_evaluation.json
python scripts/ablate_3rscan.py --manifest "$PAIR_MANIFEST" \
  --run /workspace/spatialmemory-3rscan-reviewed-final-20261010 \
  --output reports/3rscan_ablations.json
python scripts/validate_neural_depth.py --manifest "$PAIR_MANIFEST" \
  --run /workspace/spatialmemory-3rscan-reviewed-final-20261010 \
  --output reports/3rscan_neural_depth_validation.json
node scripts/check_3rscan_viewer.cjs \
  /workspace/spatialmemory-3rscan-reviewed-final-20261010 reports/3rscan_viewer_check.json
python scripts/diagnose_object_association.py \
  --run /workspace/spatialmemory-3rscan-reviewed-final-20261010 \
  --output reports/3rscan_object_association.json
/workspace/spatialmemory-clean/bin/python -m pytest -q
```

Private outputs under `/workspace/spatialmemory-3rscan-reviewed-final-20261010`:

- `reconstruction/{before,after,aligned_scene}.ply`: actual measured metric clouds.
- `changes/changes.json`: all decisions UNKNOWN; `changed_regions.ply` has no verified changes.
- `evidence/supporting_frames/`: actual detector/segmenter supporting views and masks.
- `visualization/evidence_viewer.html`: rotate/zoom aligned measured clouds; toggle model labels and the **separate native ground-truth MOVED layer**. The native layer is not a detected change.
- `visualization/native_alignment_overlay.png`: native unaligned versus native source-transform alignment; refined ICP matrix and measurements are reported separately.
- `visualization/viewer_browser.png`: checked rendered viewer; `reports/3rscan_viewer_check.json` records browser checks. Direct `file://` browser navigation was blocked by environment policy, so the self-contained content was tested with Playwright `setContent`.
- `visualization/evaluation_manifest.json`: full private source-derived audit.

These are local workspace artifacts, not a public desktop preview. Serve the private visualization directory locally if your browser blocks local-file loading. Do not publish restricted scene assets or use them as unlicensed company demo content.

## Tests, limits and remaining work

All original 92 regressions are retained; **103 tests passed, zero failed**, one Starlette/httpx deprecation warning. Eleven added tests cover rigid recovery, non-overlap rejection, scaled/reflected transforms, depth projection holes, transform direction, frame selection, conservative native annotation audit, mocked metric-pipeline orchestration and partial-evaluation provenance/ambiguity safeguards. Mocked expert orchestration is unit coverage, not real neural execution evidence; the actual runs above provide that evidence. Ruff undefined/unused-name, new-file formatting and Git whitespace checks passed; see `reports/3rscan_delivery_checks.json`.

The original sparse-neural attempt failed registration. The initial sensor trial failed held-out overlap. A denser temporal-endpoint trial failed feature-rich frame reconstruction. These failures remain preserved; later quality-selected metric execution succeeds without relaxed gates.

Remaining blockers: cross-session identity association and multi-view semantic coverage; detecting the verified small 6.9 cm displacement (current movement gate is much larger); an independent reviewed full change inventory with added/removed/unchanged controls; multiple independent test pairs; calibrated probabilistic fusion and covariance; autonomous RGB-only shared registration; complete TSDF/free-space integration and the other selected architectural candidates. A* and scene-graph extensions do not demonstrate robot deployment readiness. The broader 12-stage research system is not complete.

Exact code/document/report file changes are recorded in `reports/3rscan_delivery_files.json`. The final commit hash is reported in the delivery response; it is not embedded recursively into its own commit.

### Association diagnosis

Saved-observation diagnostics found only five exact-label cross-session candidates in the four-keyframe run. DINOv2 cosine similarity ranged from 0.159 to 0.381, below the fixed 0.85 gate. No assignment therefore reached multi-view or movement classification. Compound detector phrases and limited views also prevent stable semantic identity. The movement threshold is 0.16 m before adding the observed error band, larger than the 0.069 m reviewed displacement; this sensitivity limit is preserved rather than tuned against this positive label. See `reports/3rscan_object_association.json`.

### Expanded semantic coverage result

After the two-phone-recording goal was reiterated, another actual semantic run used twelve keyframes per recording (`configs/3rscan_metric_coverage.json`) with the same validated neural geometry cache. It completed in 611.03 seconds wall time, 602.82 seconds inference time, peak individual child RSS 3,824.49 MiB. All 69 decisions are UNKNOWN (11 geometric and 58 semantic). Eighty same-label candidate pairs all failed the unchanged 0.85 appearance gate; zero assignments were accepted. MOVED support remains one, recall/F1 0 and precision undefined. More semantic views did not resolve identity matching.

Its outputs and browser-checked viewer are under `/workspace/spatialmemory-3rscan-coverage-20261010`, with the same artifact layout as the earlier run. See `reports/3rscan_coverage_evaluation.json`, `reports/3rscan_coverage_association.json`, `reports/3rscan_coverage_ablations.json` and `reports/3rscan_coverage_viewer_check.json`. Reproduce with the coverage config and a new profiling output directory; review/ablation commands accept that directory via `--run`. The earlier four-keyframe evidence is preserved, not overwritten. The product workflow requested—two phone recordings, shared 3D, detected and described physical changes—remains a target requiring successful semantic association, sensitivity validation and actual annotated smartphone-pair tests.
