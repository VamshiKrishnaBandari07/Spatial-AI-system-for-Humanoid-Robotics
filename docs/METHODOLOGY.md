# Methodology and specification review

Reviewed against the original SpatialMemory-X Ultra engineering directive and canonical input specification on 2026-10-10. This review describes the current implementation; it does not certify completion of the full research proposal.

| Requirement | Current implementation | Remaining work or validation |
| --- | --- | --- |
| Two ordinary RGB recordings; optional sensors | Typed manifests, direct CLI paths, validated video timestamps, optional metadata readers | Supplied depth and IMU are not a complete sensor-fusion pipeline |
| Reconstruct each session | Sparse SIFT/essential/PnP triangulation; optional pinned VGGT | Full camera bundle adjustment, SLAM loop closure and dense TSDF fusion |
| Shared coordinate frame before changes | Foreground-excluded Sim(3), one-to-one RANSAC, held-out checks; failed alignment blocks claims | Independent real-scene registration benchmarks and covariance calibration |
| Metric scale | Source dataset metric metadata preserved; monocular output uses reconstruction units | Establish and validate scale before claiming metres |
| Semantic evidence | Grounding DINO, box-prompted SAM 2, masked DINOv2 appearance and supporting frames | Validate semantic identities on annotated real scenes; SAM 2 video memory |
| Independent multi-view consensus | Camera-view diversity and correlation groups; insufficient evidence remains UNKNOWN | Real-scene view-diversity and uncertainty calibration |
| Added, removed, moved, unchanged, unknown | Conservative candidate decisions; ambiguous identities remain UNKNOWN | Neural visibility is predicted evidence, not measured free-space verification |
| Uncertainty and adaptive verification | Explicit evidence gates, raw scores and optional supervised likelihood fitting | Held-out deployment calibration and adaptive evidence acquisition |
| Persistent spatial memory | SQLite observations and immutable snapshots; geometric relationships | Full 4D occupancy, orientation, dynamic graph and uncertainty propagation |
| Navigation | Experimental standalone grid A* | Validated floor map, robot integration and safety evaluation |
| Evaluation | Ground-truth-required CLI; synthetic controls and regression checks | Real smartphone before/after annotations, benchmark splits and ablations |
| User workspace | Uploads, real processing stages, 3D layers, evidence, reviews, exports and diagnostics | Multi-user authentication and shared review storage |
| Dataset integration | Verified ARKitScenes and 7-Scenes ingestion; partial Replica appearance asset | 3RScan access approval; complete Replica geometry subset |

## Verified evidence

The current Python suite passes 79 tests. Browser checks in `scripts/check_testing_console.cjs` and `scripts/check_console_edge_cases.cjs` pass using a completed synthetic run plus explicitly simulated UI failures. They verify rendering, evidence loading, exports, review persistence, theme selection, responsive controls and race/error handling. They do not measure model accuracy.

The completed synthetic neural run shown in the workspace contains 4,204 scene points, 48 registration inliers and 9/9 held-out inliers. It produced two estimated UNCHANGED object regions and no physical changes in 117.5 seconds. These figures describe one CPU run on identical synthetic panel recordings; they are not real-world accuracy, latency guarantees or comparative efficiency results. Raw detector scores are not calibrated probabilities.

The bounded research configuration selects two frames per recording. Caching validates source hashes, preprocessing, configuration and pinned model provenance before reuse. These controls bound work and prevent stale reuse; performance improvements have not been established by comparative benchmarks.

## Result integrity

UI summaries are deterministic presentations of computed classes and estimated labels. Original report explanations, measurements and model provenance remain available in the full record and unchanged report export. Human assessments are separate browser-held review records, not inference output or ground truth. UNKNOWN is not counted as a verified physical change. Failed stages record diagnostics rather than fabricated point clouds or detections.

Model weights and datasets retain their original licences. In particular, the VGGT profile uses a noncommercial research checkpoint; the interface displays this condition when selected. Raw recordings, checkpoints, caches and credentials are excluded from Git.

See [evaluation evidence](../EVALUATION.md), [implementation review](REVIEW.md), [dataset inventory](datasets_report.json), [model provenance](../MODEL_REGISTRY.md) and [input contracts](INPUTS.md).

## Dataset recovery continuation

Provenance-aware annotation schemas, a separate human annotation workspace, pair/input-hash verification, explicit benchmark identity mapping and measured process-resource profiling are now available. The current regression suite passes 92 tests. One actual ARKit recording passed a full inference control, with an UNKNOWN object result; it is not a genuine two-visit benchmark. See [dataset quality report](../reports/dataset_quality_report.md). The advanced model-stack and real-accuracy limitations above remain unchanged.

## 2026-10-10 — Real 3RScan metric validation update

The later approved real-pair run supersedes earlier unavailable-registration statements for this pair only. The original failure was zero lifted cross-session matches. An opt-in calibrated RGB-D path uses the verified native column-major metre transform and robust, gated coarse-to-fine ICP; held-out overlap is 46.09% and RMSE 0.03970 m. A fresh VGGT/semantic run completed; the final evidence run reuses integrity-validated VGGT predictions and invokes semantic experts again.

The native-source audit supports one positive MOVED instance; other categories remain UNKNOWN/unscored. It was missed: recall 0, F1 0, precision undefined. All 28 final events are UNKNOWN and no semantic association was accepted. Full inventory accuracy, object matching accuracy and calibrated posterior uncertainty are unavailable. Saved-observation ablations remain entirely abstaining. No generalization or completed research-system claim follows. The regression suite now passes 103 tests with one dependency deprecation warning.

See [exact diagnostics, commands, artifacts and limitations](3RSCAN_VALIDATION.md). Historical runs and failed diagnostics remain preserved. No model architecture, training or checkpoints were replaced.

A later twelve-keyframe-per-recording coverage run also completed (611.03 s wall time), producing 69 UNKNOWN decisions and zero accepted cross-session associations. All 80 same-label candidate pairs failed the unchanged 0.85 appearance gate. MOVED subset recall/F1 remains 0 on one supported native instance; extra views did not establish reliable detection. See `reports/3rscan_coverage_evaluation.json` and the validation instructions.

## Semantic association recovery — 10 October 2026

The full audit of the saved 69 UNKNOWN decisions identifies 50 appearance-gate rejections, 8 missing exact-label candidates and 11 geometric differences without positive free-space evidence. Actual DINOv2 replay reproduced eight stored embeddings; mask/background context changes their features but no independently reviewed identity distribution supports lowering the 0.85 floor. Configurable multi-cue assignment adds registered distance, observed extent and independent-view support, explicit unmatched options and ambiguity protection. Scores are ranking affinities, not calibrated probabilities. Bounded movement abstains when depth/alignment error budgets are absent; unresolved identity hypotheses block conflicting added/removed classifications. Legacy behavior remains the default.

A repeated real-pair run with actual semantic expert invocation and validated VGGT reuse completes with the exact previous metric alignment and 28 UNKNOWN decisions. One supported MOVED instance remains missed (recall/F1 0, precision undefined); full matching/detection accuracy is unavailable. The new standard export includes a self-contained 3D scene/evidence viewer. All 120 tests pass, including original 103. Genuine phone recordings and independent identity/error-budget review remain required; the challenge is not complete. See [HUMANOID_SUBMISSION.md](HUMANOID_SUBMISSION.md), `reports/semantic_recovery_delivery.json` and the UNKNOWN diagnostics.

## Association and abstention audit

Each semantic run exports `diagnostics/association.json` before persistent identity rewriting. Every cross-recording candidate records normalized-descriptor cosine, raw descriptor norms, semantic compatibility, registered centroid distance, partial observed extents, independent views and available raw mask scores. The same assignment solver records exact appearance, localization, view, score, ambiguity and unmatched gates; diagnostics do not relax the appearance floor or turn affinities into probabilities. Object visibility records retain occupied/free/unknown evidence in the proper registered coordinate frame.

Unknown decisions include their actual blocking gates, including missing localization, inadequate views, unresolved identity, insufficient positive free space and absent validated error budgets. The two rule-accepted assignments in the identical Chess control are not independently verified matches; all physical-change decisions remain Unknown. See the application verification report for measured execution and remaining calibration requirements.
