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
