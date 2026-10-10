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
