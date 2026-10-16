# SpatialMemory-X Ultra progress

Updated 2026-10-09. Work started in an empty unborn Git checkout. All implementation files are local; no push/publication was performed.

## Current milestone

Canonical input contracts/readers, CPU sparse/VGGT geometry, integrated detection/masks/appearance/view consensus, predicted visibility, SQLite memory and an evidence dashboard. Five pretrained CPU models have functional synthetic evidence. Clean CPU regression passes 53 tests. A full browser neural/semantic upload completed in 144.9 seconds with zero physical changes on identical synthetic videos. This is not the completed Ultra system.

| Phase | Status and evidence |
|---|---|
| 0 workspace audit | Complete: five CPU cores, 33 GiB RAM, Python 3.12.14, FFmpeg 7.1.5, no exposed NVIDIA GPU, initial empty checkout |
| 1 architecture / model registry | Implemented docs/contracts/config/CPU environment. Official source URLs and content digests audited; four public checkpoint revisions/licenses verified; hosting access restored |
| 2 video / canonical input | Implemented and tested: timestamps, VFR, orientation, hashes, quality, schemas, optional readers, explicit dataset pair conversion. Real smartphone test pending |
| 3 neural geometry | Pinned official VGGT source/checkpoint installed and SHA-256 verified; two-frame CPU inference and synthetic two-video registration/no-change pipeline passed. Real accuracy and larger workload profiling pending |
| 4 classical geometry | Executable SIFT/essential/PnP/triangulation and fixed-camera point refinement pass synthetic video integration. Full camera BA/COLMAP comparison outstanding |
| 5 registration | Sim3 RANSAC and failure gate tested with outliers/degeneracy and synthetic video. Detected-foreground exclusion and held-out correspondence gating implemented; ICP, independent scene validation and covariance outstanding |
| 6 geometry changes | Sparse candidates implemented; UNKNOWN without verified free space. Synthetic no-change passes |
| 7 semantics | DINOv2, Depth Anything V2 and Grounding DINO actual CPU inference passed. SAM 2 image masks and masked DINOv2 descriptions integrated with registered object decisions; model labels and real accuracy unverified |
| 8 temporal tracking | LK plus dense forward/backward mask consistency tested; fresh SAM 2 image masks integrated; SAM 2 video-memory propagation outstanding |
| 9 world memory | SQLite observations, both immutable snapshots and ambiguity-preserving appearance associations integrated |
| 10 consensus | Distinct-view/rule consensus integrated; optional labeled Beta-Bernoulli fitting and correlation-aware fusion implemented; real held-out calibration unavailable |
| 11 visibility | Measured-depth projection unit tested; registered VGGT predicted-depth object probes integrated and explicitly unverified; dense measured fusion outstanding |
| 12 adaptation | Bounded novelty ranking implemented; targeted neural re-inference and information-gain calibration outstanding |
| 13 full change reasoning | Semantic class/identity/view gates integrated and unit tested; actual model no-change control passed; MOVED/ADDED/REMOVED real-video validation outstanding |
| 14 descriptions | Deterministic geometric/object explanations, raw scores and supporting masks integrated; validated free-form narratives outstanding |
| 15 application | Local upload, actual stages, 3D point/camera/object layers, mask evidence, filters, scene memory and registration panel; browser/reload/mobile smoke passed. Real benchmark panel data unavailable |
| 16 navigation | Grid A* and inflation tested independently; floor geometry integration outstanding |
| 17 benchmark | Annotation/scenario templates and label metrics implemented; no real data or measured benchmark |
| 18 optimization | Full-duration bounded frame selection, sparse Jacobian, single worker, output lock and provenance-checked geometry reuse implemented; comparative profiling outstanding |
| 19 docs / demonstration | Actual pretrained synthetic browser demonstration and desktop/mobile screenshots saved; real recording demonstration outstanding |
| 20 final verification | CPU installation/tests verified; complete acceptance, Docker, public GitHub and new-task restoration unrun |

## Diagnosed failures corrected

1. LK pyramids were too coarse on small images; size-aware levels restored valid camera-translation tracks.
2. Clean CPU tests imported optional torch too early; unloaded-model diagnostics now work without neural extras.
3. OpenCV disabled automatic orientation by default; ingestion explicitly enables orientation correction. MOV display-rotation tests pass.
4. Current FFmpeg requires `-display_rotation` for the rotation test fixture; metadata-only `rotate` did not create a display matrix and was corrected.

## External prerequisites

- Model-host access is restored. Updated reusable neural installation/start instructions are saved separately from runtime execution/publication.
- Supply two actual recordings or valid licensed dataset pairs. Supervised metrics additionally require real annotations.
- Public checkpoints require no token and downloaded successfully. The original VGGT checkpoint is noncommercial research only; gated commercial weights remain optional and subject to their terms.

## Configuration

Cloud draft saved model-host domain additions, the tested complete CPU install_script and app start_skill. Review/save and publication are user actions; neither is claimed complete. Current files/dependencies do not imply live processes survive publication.

## Neural continuation after settings update

Public hosting now responds successfully. Three small experts passed real CPU inference on a synthetic image. VGGT-1B (verified SHA-256, reviewed official source) passed a two-frame CPU smoke in 72.8 seconds at 9,883 MiB peak process RSS. The integrated VGGT two-video synthetic control reconstructed 2,102 surface feature points, aligned 1,456 correspondences and returned zero candidates in 74.9 seconds. These figures are functional smoke evidence, not real-world accuracy or an ablation comparison. `scripts/install_neural.sh` ran successfully and pip check passed.

Live VGGT application upload validation passed: real pretrained geometry outputs reached the report and interactive world endpoint, zero candidates on the synthetic identical control. Neural install_script/start_skill updates are confirmed saved in the environment draft; publication remains a product/user action and is not claimed. CPU regression suite now passes all 53 tests and pip check reports no broken requirements.

Further corrected failures: NumPy bounding-box coordinates needed native integer serialization; the failed run was retained and a new completed run verified the fix. Long-video frame budgets now span the entire original recording. Model revisions must be immutable 40-character commits. Concurrent writers are rejected. Geometry cache reuse verifies exact source/preprocessed frames before accepting predictions. Documentation reflects implemented semantic integration without claiming real accuracy, calibrated probabilities or full Ultra completion.

Dataset-only continuation: two ARKitScenes visits and one selected 7-Scenes recording are downloaded and prepared, with scene-disjoint manifests and seven explicitly synthetic geometric-region pairs. Replica has a partial atlas asset; 3RScan access approval remains required. Real data loaders pass; 67 regression tests pass. No architecture, checkpoint or training changes were made for this dataset work. See docs/DATASETS.md and docs/datasets_report.json.

## 2026-10-10 — Testing console review

Verified and corrected UI persistence, trial-load races, draft handling, upload failure recovery, camera/depth validation, changed-region exports and atomic report completion. 79 regression tests and desktop/mobile plus ten browser edge-case groups pass. See docs/REVIEW.md. Historical runs, datasets, checkpoints and training were preserved.

## 2026-10-10 — Professional workspace and publication review

Restored SpatialMemory-X Ultra branding, direct user wording, dark/light themes, readable processing stages and evidence summaries while preserving original report provenance. Both browser suites and all 79 Python tests pass. Added methodology/specification status review; no model architecture or training changes.

## 2026-10-10 — Dataset recovery and ground-truth safeguards

Verified actual on-disk sizes, seven retained archives and 19,531 CRC-checked payload files. Validated all 144 prepared native frames, per-frame ARKit calibration, source timestamps, metric depth conversion and camera/world round trips; generated ignored native ARKit depth clouds. Replica remains appearance-only and 7-Scenes depth is not silently registered with RGB.

Researched official 3RScan/RIO10 documentation and ETHZ/TU Wien alternatives. Access forms remain unsubmitted; official host additions were saved without changing model architecture, checkpoints or training. No accessible, licensed, complete real annotated before/after pair obtained. Added typed annotation levels/provenance, an independent annotation workspace, source-hash pair validation, explicit identity mapping, provenance-gated benchmark arithmetic and process resource profiling.

Actual full pretrained RGB inference on a single real ARKit recording supplied twice completed in 171.7 seconds, 9,961 MiB peak individual child RSS; 1,348 points per state, 761 registration inliers and 152/152 held-out inliers. One estimated electronic-device region remains UNKNOWN with one distinct view per visit; semantic label unverified. This is an inference control, not supervised real-scene accuracy.

Development and independent clean CPU suites each pass 92 tests. Annotation-browser and existing edge-case browser checks pass. Reports retain rejected pairing, unavailable metrics, acquisition restrictions and explicit incomplete ten-condition status. Genuine changed-pair registration, supervised results and ablations remain blocked.

### 2026-10-10 — Approved real 3RScan pair acquired

After confirmed official approval and Terms of Use agreement, one native training reference/rescan pair was downloaded from the official HTTPS host with bounded transfers and SHA-256 journals. Both ZIPs were fully read with CRC validation. All 257 RGB-D/pose frame sets and 40,097 annotated vertices passed native checks. Existing canonical adapter conversion and metadata validation passed. Together with earlier checks, 401 native frame sets have been inspected. Native change IDs are retained in an UNREVIEWED canonical draft; no reviewer identity or extra change labels were invented.

A fresh unchanged VGGT/semantic consensus run on the pair failed at shared registration after 346.2 seconds. Both visits produced finite neural arrays; alignment had insufficient nondegenerate consensus. No registration thresholds were relaxed and no change predictions, accuracy or F1 were fabricated. Full completion remains false. See reports/3rscan_acquisition.json, reports/3rscan_native_validation.json and reports/3rscan_inference_attempt.json.

## 2026-10-10 — Real 3RScan metric validation update

The later approved real-pair run supersedes earlier unavailable-registration statements for this pair only. The original failure was zero lifted cross-session matches. An opt-in calibrated RGB-D path uses the verified native column-major metre transform and robust, gated coarse-to-fine ICP; held-out overlap is 46.09% and RMSE 0.03970 m. A fresh VGGT/semantic run completed; the final evidence run reuses integrity-validated VGGT predictions and invokes semantic experts again.

The native-source audit supports one positive MOVED instance; other categories remain UNKNOWN/unscored. It was missed: recall 0, F1 0, precision undefined. All 28 final events are UNKNOWN and no semantic association was accepted. Full inventory accuracy, object matching accuracy and calibrated posterior uncertainty are unavailable. Saved-observation ablations remain entirely abstaining. No generalization or completed research-system claim follows. The regression suite now passes 103 tests with one dependency deprecation warning.

See [exact diagnostics, commands, artifacts and limitations](docs/3RSCAN_VALIDATION.md). Historical runs and failed diagnostics remain preserved. No model architecture, training or checkpoints were replaced.

A later twelve-keyframe-per-recording coverage run also completed (611.03 s wall time), producing 69 UNKNOWN decisions and zero accepted cross-session associations. All 80 same-label candidate pairs failed the unchanged 0.85 appearance gate. MOVED subset recall/F1 remains 0 on one supported native instance; extra views did not establish reliable detection. See `reports/3rscan_coverage_evaluation.json` and the validation instructions.

## Semantic association recovery — 10 October 2026

The full audit of the saved 69 UNKNOWN decisions identifies 50 appearance-gate rejections, 8 missing exact-label candidates and 11 geometric differences without positive free-space evidence. Actual DINOv2 replay reproduced eight stored embeddings; mask/background context changes their features but no independently reviewed identity distribution supports lowering the 0.85 floor. Configurable multi-cue assignment adds registered distance, observed extent and independent-view support, explicit unmatched options and ambiguity protection. Scores are ranking affinities, not calibrated probabilities. Bounded movement abstains when depth/alignment error budgets are absent; unresolved identity hypotheses block conflicting added/removed classifications. Legacy behavior remains the default.

A repeated real-pair run with actual semantic expert invocation and validated VGGT reuse completes with the exact previous metric alignment and 28 UNKNOWN decisions. One supported MOVED instance remains missed (recall/F1 0, precision undefined); full matching/detection accuracy is unavailable. The new standard export includes a self-contained 3D scene/evidence viewer. All 120 tests pass, including original 103. Genuine phone recordings and independent identity/error-budget review remain required; the challenge is not complete. See [docs/HUMANOID_SUBMISSION.md](docs/HUMANOID_SUBMISSION.md), `reports/semantic_recovery_delivery.json` and the UNKNOWN diagnostics.

### Application verification — 11 October 2026

Fresh UI uploads invoke actual neural models and complete reconstruction, registration, semantic processing, visualization and export. Per-candidate association diagnostics, exact Unknown blocking checks, result ZIP downloads and incomplete-upload cleanup are integrated. 129 tests and browser/API checks pass. The idle application backend was updated on port 8000; prior completed runs remain accessible. The real Chess identical-recording control returns 48 Unknown decisions, and independently verified identities/calibration remain unavailable. See `reports/APPLICATION_VERIFICATION.md` and `docs/PHONE_DEMO.md`; the original smartphone change-detection challenge is not complete.

## 2026-10-16 — Engineering hygiene pass (no model or accuracy change)

Verified in a clean sandbox (no GPU, no model weights, no private recordings): a fresh virtualenv built from `requirements/cpu.lock.txt` now passes all 344 tests with `torch` absent.

Defects found and fixed:

1. The documented clean-CPU install and CI failed at test collection: five modules and two tests import Pillow, which was only in the neural lock. Pillow is now pinned in `cpu.lock.txt` and the `test` extra; `kornia` (needed by the LoFTR matcher, already in `neural.lock.txt`) is now also in the `neural` extra.
2. `LearnedCropMatcher` imported torch/kornia before verifying the checkpoint checksum, so an unverified file raised `ModuleNotFoundError` instead of the checksum error without neural extras. The hash is now streamed and checked first.
3. Core loaders, scripts and installers hardcoded `/workspace/...`. All locations now resolve through `spatialmemory.paths` with environment overrides; defaults are unchanged. A test forbids new hardcoded paths in the package.
4. A corrupt upload surfaced a raw `CalledProcessError` including the ffprobe command and server path. Ingestion now raises a clear `ValueError` naming only the file.
5. `group_with_gallery` retained observations whose identity conversion failed but discarded the reason; it is now recorded as `conversion_error` in provenance.
6. Two blind `pytest.raises(Exception)` assertions now name the expected exceptions; CI gained a lint step (undefined names, unused imports).

Exercised for real with the synthetic fixture on the CPU sparse backend: upload, run, report, world, result ZIP, concurrent-job rejection (HTTP 429) and path-traversal rejection through the live FastAPI app. This is a functional check, not a measure of accuracy, and it does not verify the neural workflow or a hosted deployment.

Not addressed: every P0/P1 item in `docs/CLAUDE_HANDOFF.md` needs the private phone recordings, model weights and/or a GPU, none of which were available. No detection, registration, identity or visibility behavior was changed, and no accuracy claim follows from this pass.

### Follow-up (same day): investigation of the AFTER geometry failure (handoff item 1)

Not resolved; the private run artifacts were unavailable. What was established without them:

* `pixel_mapping.native_to_geometry_matrix` was checked against a torch-free re-implementation of the pinned upstream `load_and_preprocess_images` (commit a288dd0). It agrees to 0.025 model pixels for 10 phone/landscape/square shapes in `pad` and `crop` modes. Mutation checks (dropping the pixel-centre term, ignoring padding, flipping the crop sign) fail 19, 9 and 5 of 30 cases respectively. A native/model coordinate-mapping error is therefore unlikely to explain the failure.
* The 2-native-pixel gate is a point-head versus camera-head self-consistency test in the same index convention upstream uses. Expressed in VGGT input pixels it is only about 0.54 px for 1080x1920 frames. The audited median of 14.57 native px corresponds to roughly 4 model px at that scale (assumes a 1080x1920 frame; the real frame size was not available). Whether that is a systematic shift, a focal disagreement or unstructured head disagreement is undetermined.
* The threshold was not changed. Added `spatialmemory.head_consistency` to decide this from the saved run:
  `python -m spatialmemory.head_consistency --run <run_dir> --output heads.json`
  It reports per-view, per-role residuals with frame ids and decomposes them into shift, focal and unstructured parts. Validated only on synthetic predictions with injected errors.

## 2026-10-16 — Reproduction audit against the handoff (no private artifacts available)

All 14 private paths listed in `docs/CLAUDE_HANDOFF.md` were absent from the audit machine (1 CPU, ~3 GiB RAM, no GPU, Hugging Face unreachable), so nothing requiring recordings, weights or saved runs could be re-run. Reproduced from committed material only:

| Handoff claim | Result |
|---|---|
| "333 passed" | Not reproducible on the documented clean CPU install at `fc66dc3`: pytest stops with 3 collection errors (Pillow missing). Fixed earlier; 416 tests now pass in a clean venv. The 333 figure held only in the private venv. |
| Both local VLMs yield zero accepted proposals | Reproduced at the validation layer by replaying the raw outputs stored in `reports/local_structured_visual_validation.json` through the repo validator: 500M echoes a frame header (not JSON); ~2B returns JSON with forbidden keys and invented frame IDs (`"1234567890"`). Prompt recency was ruled out (the instruction already follows the images). Model quality, prompt wording and unconstrained decoding are not separable without weights. |
| Removal not promoted without free-space evidence | Confirmed for the production path (`semantics.py`: needs multiple distinct views, at least 2 supported free views, and no same-label unmatched candidate in the other recording). `pipeline.py` calls `compare()` without a free-space callback, so it cannot promote. |
| AFTER geometry inconsistent (61/64 rays, median 14.57 px) | Not reproducible without the saved run. Coordinate mapping ruled out as the cause (see above). |

Defects found and fixed in the reference decision rule (`change.classify_object`, used by tests rather than the live pipeline): NaN positions returned UNCHANGED, infinite ones MOVED, a non-finite threshold UNCHANGED, any truthy non-boolean evidence (dict, string) promoted ADDED/REMOVED, and two absent observations with free-space evidence returned ADDED. It now fails closed (UNKNOWN) and rejects invalid configuration. `compare()` applies the same literal-True rule to its free-space callback. Local VLM failures now carry a `failure_category` (`not_json`, `schema_violation`, `invented_frame_reference`) so candidate models can be compared by failure mode.

No detection, registration or identity behaviour changed. No accuracy claim follows.

### Default storage location fixed for ordinary users (same day)

Running the documented `uvicorn spatialmemory.app:app` command as a normal user crashed with `PermissionError: /workspace/spatialmemory-runs`, because the earlier path change still defaulted to `/workspace`. Defaults are now: an explicit environment variable, else `/workspace` when it exists and is writable (existing installs unchanged, including read-only preinstalled models), else `~/.spatialmemory`. `.env.example` no longer forces `/workspace`.

## Uploaded ZIP integration review — 2026-10-16

Integrated the submitted fixes and corrected residual scaling and invalid centroid validation. Added regression tests; 425 passed, zero failed, one deprecation warning. Shell syntax and diff whitespace checks passed. Diagnostic replay of existing real-video predictions still shows AFTER geometry inconsistency (after:2040 median 38.04 native pixels). This is not a new inference run and does not establish removal detection. See reports/claude_zip_integration_review.json.
