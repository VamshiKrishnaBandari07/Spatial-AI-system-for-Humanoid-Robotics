# Validation evidence

## Executed checks (2026-10-09)

The final clean CPU environment ran **53 tests: 53 passed**, with one Starlette/httpx deprecation warning. A fresh virtualenv was installed from `requirements/cpu.lock.txt`, the package installed without inherited site packages, and `pip check` found no broken requirements. This validates the CPU install script in the current cloud machine; it does not establish fresh cloud snapshot restoration or Docker execution.

Tests exercise Sim(3) alignment with 20% injected outliers, degenerate registration rejection, no-change points, positive-free-space gating for additions/removals, movement identity safeguards, unknown/occupied/free/occluded projection, correlation-group fusion, bounded frame ranking, unknown-blocked A*, input integrity, corrupt video failure, synthetic video reconstruction, API validation, path escapes, optional metadata fallback, camera rescaling, annotation scale checks, pose arrays/quaternions/rotation matrices, depth formats, IMU metadata, ScanNet binary decoding, dataset-pair rejection, native 3RScan conversion, immutable memory, identical-object ambiguity, camera-compensated flow, measured label metrics, original VFR timestamps, MOV orientation and sensor synchronization errors.

## Live application functional check

A real HTTP upload request to the running local application processed two **synthetic identical videos**, reconstructed 1,764 before points, aligned 1,519 3D correspondences and produced zero geometric candidates. Recorded runtime was 3.146 seconds on this machine. A world-array request and an evidence overview-frame request succeeded. The identity alignment residual is approximately numerical roundoff because the recordings are identical; it is NOT an estimate of real-world registration accuracy. Raw current-instance evidence is in ignored `results/app_smoke.json` and local job output. No private videos were used.

## Not measured

No annotated real smartphone before/after change benchmark is available. Verified ARKitScenes and 7-Scenes recordings are available for ingestion and reconstruction checks, but do not supply the required object-change ground truth. Real-world precision, recall, F1, false removal/addition rates, matching accuracy, 3D localization error, coverage, GPU memory and ablations are **unrun**, not zero and not passing. DINOv2, Depth Anything V2, Grounding DINO and VGGT passed pretrained CPU inference smoke checks, but their accuracy is not validated with real ground truth. Docker build/runtime validation is blocked by container DNS; host package-server connectivity works.

Final acceptance tests A–H and N require real video/model outputs and remain unrun. Tests I/J have synthetic/unit evidence only. K has sparse fit diagnostics, not calibrated uncertainty. L has clean CPU installation evidence. M has ignored private-data/checkpoint paths and no uploaded private recordings; the source repository is published on GitHub, while no completed research benchmark submission is claimed.

## Real evaluation procedure

Capture overlapping slow scans of the same room before/after a controlled edit. Annotate object IDs and change classes independently; annotate visibility and positions only when actually measured. Include unchanged controls, occlusion and partial coverage. Use `benchmarks/scenarios.json` and `examples/annotations_template.json`. Never populate a template with illustrative coordinates and call it ground truth.

Run inference without annotations, retain provenance and diagnostics, then run `python -m spatialmemory_x evaluate --manifest ... --annotations ...`. The current evaluator is object-ID-based classification over annotated identities; absent predictions are UNKNOWN. It does not yet report exhaustive false-positive events, localization or region IoU, and sparse geometric candidates have no semantic object IDs. The research profile writes persistent local object identities; associating those identities with external ground-truth IDs remains a dataset evaluation prerequisite. This is an evaluation tool interface, not a demonstrated semantic benchmark.

Ablations are deferred until real data, model outputs and full integration exist. Do not interpret standalone synthetic module tests as an ablation study.

## Actual pretrained inference and neural geometry continuation

After the settings update, Hugging Face access worked. DINOv2 produced a finite normalized 384-dimensional descriptor; Depth Anything V2 produced a finite 518×686 relative-depth array; Grounding DINO produced a table label at raw score 0.326 on a synthetic panel image, which is not verified semantic evidence. VGGT generated finite camera, intrinsic, depth and point-map arrays for two synthetic frames. Model loading/inference took 72.8 seconds and peaked at 9,883 MiB process RSS. The integrated two-video VGGT synthetic identical control ran in 74.9 seconds, with 2,102 surface feature points, 1,456 registration inliers and zero candidates. These metrics are smoke-test measurements on this machine, not smartphone performance. Exact summaries and pinned revisions are in docs/neural_smoke.json, docs/vggt_smoke.json and docs/vggt_pipeline_smoke.json.

The live application also passed POST /jobs with geometry_backend=vggt, job completion, report/world requests and the synthetic no-change assertions. Its measured runtime was 94.7 seconds under concurrent validation load, with 2,102 points and 1,456 registration inliers. Evidence: docs/vggt_app_smoke.json. This confirms HTTP integration, not real-world change-detection accuracy.

## Integrated semantic and browser validation

The full browser upload used actual pinned VGGT, Grounding DINO, SAM 2 and DINOv2 inference on synthetic identical videos. It completed in **144.9 seconds**, with two proposed UNCHANGED object regions, zero physical changes, 48 registration inliers and 9/9 held-out correspondence inliers. The detector called synthetic panels “electronic device”; those labels are unverified. The identical-control residual is numerical roundoff, not real registration accuracy.

Browser checks passed mask-image loading, object selection, layers, zoom, retained-run reload, mobile overflow and absence of JavaScript errors. Screenshots and `docs/browser_smoke.json` preserve the evidence. A separate geometry-cache semantic run completed in 35.3 seconds; it reused validated neural arrays and is not comparable to full inference runtime. The first semantic run failed NumPy-integer JSON serialization; conversion to native integers corrected it, and the fresh successful run supplied the acceptance evidence.

The 53-test suite additionally covers viewpoint deduplication, identical-object ambiguity, conservative visibility consensus, registered stable scene memory, optical-flow mask consistency, foreground exclusion, held-out registration, training-likelihood group deduplication, exclusive output locking, immutable revision enforcement, cache/input mismatch rejection, evidence-path guards and full-duration frame budgeting. Ruff undefined/unused-name checks and formatting checks pass for Python source, scripts and tests. Neural installation repeated successfully and `pip check` passed. These checks do not substitute for a real before/after benchmark.

## Dataset-only continuation

Official ARKitScenes and 7-Scenes RGB-D subsets are now locally verified; Replica has a verified partial texture asset. The regression suite now passes 67 tests. Real before/after object-change ground truth remains unavailable, so prior accuracy limitations still apply. Native loaders and seven clearly synthetic geometric-region pairs were validated without changing models or starting training. See [dataset evidence](docs/DATASETS.md).

## Current regression and publication status (2026-10-10)

At that review, both the development environment and independent clean CPU environment passed all 79 tests. The professional workspace and edge-case browser suites pass. The source repository is published, with recordings and model weights excluded. Historical test counts and runtimes above describe their original runs, not the current total. See [methodology review](docs/METHODOLOGY.md) for current scope.

## Fresh full-pipeline prediction check (2026-10-10)

Actual CPU inference with the consensus configuration and no geometry reuse completed in 182.9 seconds on one identical synthetic recording pair. It returned two estimated UNCHANGED regions, zero physical-change alarms and 2,102 finite points per recording. Registration passed with 48 inliers and 9/9 held-out inliers. Supporting image files were verified. This is a functional no-change control, not an accuracy benchmark. The reciprocal of this one run is approximately 0.33 pairs/minute, including initialization; sustained throughput and real-world accuracy remain unmeasured. Evidence: [prediction check](docs/prediction_check_20261010.json).

## Dataset recovery validation (2026-10-10)

The updated development and independent clean CPU suites each pass **92 tests**. All 144 prepared native frames and 19,531 archive payloads were revalidated. Annotation import/export, independent-review checks, preview handling and mobile layout pass browser checks. The new `scripts/benchmark_scene.py` rejects the unestablished ARKit visit pairing and computes no supervised metrics.

Fresh full pretrained inference on actual ARKit recording 47333462, supplied twice as a loader control, completed in 171.7 seconds (173.1 seconds including subprocess startup). Peak individual child RSS was 9,961.1 MiB. It reconstructed 1,348 points per state and returned one UNKNOWN region with an unverified electronic-device label; only one distinct object view existed per visit. The alignment gate passed with 761 inliers and 152/152 held-out inliers, but identical-input residuals cannot establish cross-visit accuracy.

See [real-recording control evidence](reports/real_recording_inference_control.json), [quality report](reports/dataset_quality_report.md) and [annotation protocol](docs/annotation_protocol.md). No genuine real changed pair or independent change labels were obtained. Real F1, precision/recall, localization, matching accuracy, uncertainty calibration and ablations remain unrun. The legacy `evaluate` command performs exact-ID label arithmetic; the provenance-gated script is required for a verified real benchmark.
