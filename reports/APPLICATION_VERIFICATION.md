# Application verification — 11 October 2026

The application is operational as a local research prototype. This is not completion of the real smartphone challenge or proof of reliable physical change detection. [Measured results](application_verification.json) and [test log](application_tests.txt) distinguish execution from accuracy.

| Component | Status | Test evidence | Remaining issue |
|---|---|---|---|
| Frontend | PARTIAL | Fresh UI upload; progress, actual scene/evidence, downloads; desktop/mobile and negative browser checks | Some codecs cannot preview in Chromium; explicit message and extracted-image fallback. Genuine phone interaction untested |
| Backend | PASS | Fresh API job completes; actual busy/empty/corrupt requests; preserved reports; 129 tests | Local single-worker evaluation service, not authenticated multi-user deployment |
| Video processing | PASS | Actual Chess ingestion and frame extraction; existing corrupt, variable-rate and orientation regressions | Genuine phone recordings unavailable |
| Neural reconstruction | PARTIAL | Fresh VGGT invocation on four frames per recording; finite exported clouds | Monocular scale and real geometry accuracy unverified; sparse keyframe coverage |
| Registration | PARTIAL | Identical-recording correspondence registration; real sensor-cloud replay passes original gates | Source-assisted real-pair refinement is not autonomous RGB-only cross-session registration |
| Object segmentation | PARTIAL | Actual Grounding DINO, SAM 2 and DINOv2 inference with masks/evidence | Mask and semantic accuracy not independently measured; duplicate/compound labels remain |
| Semantic association | PARTIAL | 625 audited candidates and two rule-accepted assignments; one-to-one/unmatched/ambiguity tests | Identical inputs are not independent correspondence truth; real cross-session matching remains unverified |
| Movement detection | PARTIAL | Synthetic bounded-error tests; actual matched objects abstain with exact missing-budget gate | Validated error budgets and genuine moved-object success unavailable |
| Change visualization | PASS | Actual saved clouds and Unknown regions render; supporting images and class/layer controls tested | No real confirmed moved/added/removed visualization is claimed |
| Report export | PASS | JSON, PLY, diagnostics, evidence and readable 132-member ZIP; negative path/symlink tests | Browser reviews export separately; raw videos/checkpoints excluded |
| Real-video validation | BLOCKED | Real dataset RGB control processed, not phone change benchmark | Genuine same-room phone pair, independent labels and measured edits required |

## Actual integrated run

The browser uploaded the same prepared native Chess `seq-01` recording into both slots and selected Neural reconstruction and Analyze objects. There were no mocked model outputs or geometry-cache reuse. VGGT, Grounding DINO, SAM 2 and DINOv2 executed on CPU.

- Latest inference: **322.41 seconds**; before/after clouds: **1,677 points each**.
- **48 Unknown decisions**, zero declared added/removed/moved/unchanged changes.
- **625 candidate pairs**: 492 semantic-incompatible, 108 below the unchanged appearance floor, 21 insufficient independent views, two missing localization, two accepted by assignment rules.
- The two rule-accepted assignments are not independently verified correspondences. Movement abstains because validated error budgets are missing.
- Precision, recall, F1, matching accuracy and calibration metrics remain **unavailable** for this control.
- Control registration: 1,010/1,010 inliers and 202/202 held-out inliers. Near-zero residual is expected for identical inputs, not real-world reconstruction accuracy.
- First fresh application run completed in 304.44 seconds before the diagnostic/export additions. These two durations are not a controlled efficiency comparison.

The real 3RScan registration-only replay used saved sensor clouds and the verified native initial transform. The previously aligned after cloud was restored by inverse estimated transform before replay. Original gates pass: RMSE **0.03346375 m**, held-out RMSE approximately **0.03970 m**, symmetric overlap approximately **0.4861**. RMSE differs by approximately **6.1e-9 m** from the preserved run. No new neural reconstruction or semantic benchmark is claimed by this replay. Registration implementation and gates were not changed.

## Artifacts and reproducibility

Latest private application run:

```text
/workspace/spatialmemory-app-validation-20261011/75b790772f4642e4a2bc66f728328811/output/
/workspace/application-final-e2e.zip
/workspace/application-final-e2e.desktop.png
```

The run retains configuration, seed, hardware/dependency information, original upload copies, timestamp/frame manifests, fresh neural arrays, raw semantic outputs, persistent memory, exact rejection gates, visibility records, processing stages and model/source provenance. All private dataset-derived imagery, clouds and neural artifacts remain outside Git. The ZIP passed CRC checks; required JSON/PLY/NPZ/viewer files are present and cloud arrays are finite.

Tests: **129 passed, zero failed**, preserving all prior 122. One existing Starlette/httpx deprecation warning remains. Python lint, formatting and JavaScript syntax checks pass. Browser replay covers actual outputs; busy/offline/media events in those replay checks are explicitly simulated. Separate actual API tests verify busy rejection, empty upload cleanup and corrupt-file failure. An initial preview-recovery test raced a real codec error; it now waits for the media state to settle before simulated events.

The idle application on port 8000 was restarted to load the tested backend; its four prior completed runs and new download routes were verified. No training or unrelated process was restarted. Exact launch and recording steps: [phone demonstration guide](../docs/PHONE_DEMO.md). Actual inference test: `scripts/check_application_e2e.cjs`, with explicit source paths and a new private output prefix.

## Blockers and scope

Independent source-based identity examples and changed/unchanged calibration data are missing. Bounded movement intentionally abstains without declared validated error budgets. Scores and raw mask IoUs are not probabilities or measured segmentation quality. GPU execution is BLOCKED by the CPU-only environment; no GPU fallback validation is claimed. The shipped stack does not establish full MASt3R/COLMAP cross-validation, SAM 2 video-memory tracking, independently calibrated probabilistic fusion or a grounded language-model extension. The source-assisted 3RScan test does not establish autonomous phone-video registration or generalization.

Use the empty annotation template and independent-review protocol; do not turn model suggestions into ground truth. One annotated genuine phone pair establishes feasibility only. Multiple independent pairs and held-out calibration are required for a meaningful benchmark. Production readiness and commercial licensing eligibility are not established.
