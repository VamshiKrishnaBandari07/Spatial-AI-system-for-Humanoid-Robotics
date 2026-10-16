# Claude engineering handoff — 16 October 2026

> **Update 16 October 2026 (hygiene pass).** Infrastructure defects were fixed and
> the suite is now 344 tests; see `PROGRESS.md`. No row of the prioritized table
> below was resolved: those items need the private recordings, checkpoints and/or a
> GPU runtime. Treat the table as still fully open. For item 1, run `python -m spatialmemory.head_consistency --run <fresh run dir>` first; see `PROGRESS.md`.

## Objective and honest baseline

Finish a system that automatically compares two phone recordings, produces a
shared 3D scene, identifies added/removed/moved objects or regions, and provides
supporting images, locations and explanations. Do not restart the project or
replace working components without evidence that replacement is necessary.

Repository: https://github.com/VamshiKrishnaBandari07/Spatial-AI-system-for-Humanoid-Robotics

Baseline commit before this handoff: `2052489526a6a8a341e6819b49dff9b88eb95d53`.
Workspace: `/workspace/Spatial-AI-system-for-Humanoid-Robotics`.

**Not complete:** fresh neural inference finishes, but automatic change detection
has not passed the submitted phone-pair acceptance test. Manual visual findings
are not automated predictions. Passing regression tests is not model accuracy.

Read first:

1. `docs/SUBMISSION.md`
2. `reports/fresh_phone_validation_20261015.json`
3. `docs/PIPELINE_FAILURE_AUDIT.md`
4. `reports/removal_depth_gate_audit.json`
5. `reports/local_structured_visual_validation.json`
6. `docs/DENSE_SCENE_EXPORT.md` and `docs/LOCAL_VISUAL.md`

## What actually ran

- Fresh uncached phone inference: **992.9047 seconds**, CPU.
- VGGT geometry plus Grounding DINO, SAM 2, DINOv2 and semantic label checking.
- **53 UNKNOWN decisions; zero confirmed added/removed/moved changes.**
- Sparse Sim(3): 20/24 inliers, held-out 3/4, RMSE 0.024237 and held-out
  RMSE 0.012554 in unverified reconstruction units, not metres.
- Dense filtered export: 37,297 before points; 22,435 after points.
- Dense after-to-before fraction within the registration threshold: 17.50%.
  Partial coverage and geometry errors affect this number; it is not accuracy.
- Full regression suite: **333 passed**, one dependency deprecation warning.
- Chromium dense-viewer loading and before/after toggles: passed, no JS errors.
- Neither tested local VLM produced usable structured change findings.

## Resources and access needed

The existing workspace has no usable CUDA GPU, about 33 GiB total RAM, and about
3.2 GiB disk free at the last check. Recheck before downloading anything.

For the proposed 7B VLM experiment, provide a GPU runtime with sufficient VRAM
for weights plus multi-image activations/KV cache. A 24–48 GB GPU is a practical
starting range; actual requirements depend on precision, quantization and image
budget and must be measured. Prefer 32–64 GB host RAM and at least 30–50 GB free
storage for checkpoints, environments and artifacts. This is a recommended test
configuration, not a guarantee of success or a proven minimum.

Qwen2.5-VL-7B-Instruct was recommended as an **untested candidate**, not installed
or validated here. Check official version compatibility and license. The current
local adapter uses `Idefics3ForConditionalGeneration`: **changing its model ID to
Qwen is not sufficient**. Implement the appropriate model/processor adapter and
verify a real image request first. Pin/checksum artifacts and retain offline
inference. Do not provision paid resources or use external image APIs without
appropriate user authorization. Never ask for credentials in chat.

The user has already supplied enough recordings to debug the present failure.
Do not request duplicate recordings as a substitute for fixing implementation.
For generalization and calibrated benchmark claims, separate held-out scenes and
verified annotations will still be required. Access to this workspace or a
secure transfer of its private files is needed if Claude runs on another machine;
GitHub alone does not contain the videos, outputs or model weights.

## Prioritized unfinished work

| Priority | Problem and evidence | Relevant files | Required completion evidence |
|---|---|---|---|
| P0 | AFTER geometry inconsistent with camera rays: 61/64 sampled target rays exceed 2 native pixels; median 14.5723 px | `neural_geometry.py`, `pixel_mapping.py`, `semantics.py`, `object_geometry.py`, `reprojection.py` | Trace exact frame IDs, K, pose, preprocessing and heads; correct the earliest discrepancy; show per-view residuals and independently checked correspondences without relaxing gates |
| P0 | Whole-scene/object-level alignment remains unverified despite sparse registration acceptance | `registration.py`, `classical_validation.py`, `dense_export.py` | Static-landmark overlays, held-out object-neighborhood checks and dense overlap diagnostics; distinguish partial coverage from registration error; retain failure rejection |
| P0 | Small-object recognition fails: tight pink-container crop labeled table; rotation recovered a bottle but also false labels | `models.py`, `semantics.py`, `segmentation.py`, `semantic_labels.py`, `crop_recheck.py`, `oriented_detection.py` | Correct automatic localization and descriptive/category labeling on real before/after crops with context, including other objects and negative cases |
| P0 | Current local VLMs do not follow the comparison task reliably | `local_visual.py`, `visual_comparison.py`, `scripts/install_local_visual.py` | Real model-authored change proposals with correct role/frame references and supporting images; reject invented references and unsupported descriptions |
| P1 | Cross-recording identity and independent-view support unresolved | `association.py`, `tracking.py`, `temporal_neural.py`, `temporal_review.py`, `object_geometry.py` | Test identity switches, fragmented tracks, reacquisition and lookalikes; global one-to-one association and uncertainty remain intact; demonstrate same physical object across views |
| P1 | Removal evidence fails: 62/64 in-frame samples fail depth support, two FREE, zero supported free views | `semantics.py`, `surface_evidence.py`, `object_visibility.py`, `raybox_integration.py` | Validated expected location plus multiple distinct unobstructed observations and positive empty-volume evidence; test occluded, outside-view and unchanged negatives |
| P1 | Uncertainty is not calibrated; ray-box adapter lacks validated bounds | `calibration.py`, `detector_calibration.py`, `raybox_integration.py` | Document measurement/calibration data and propagation; held-out reliability/coverage checks; raw scores cannot be called calibrated probabilities |
| P1 | Added/moved classes not demonstrated on independent real pairs | `semantics.py`, `association.py`, evaluation CLI/readers | Verified annotations kept outside inference; per-class metrics only where labels support evaluation; movement identity and scale conventions explicit |
| P2 | UI exists but live hosted inference has not been verified | `app.py`, `dashboard.html`, `colab_ui.py`, `site/`, notebooks and hosting docs | Actual upload→run→evidence→3D→download test, authentication/error states, new-job isolation and usable deployed URL; static GitHub Pages is not a model server |
| P2 | CPU performance is unsuitable for fast interactive use | `models.py`, `segmentation.py`, `local_visual.py`, `pipeline.py` | Stage timings and memory measurements; tested GPU/device handling, bounded batching/caching; no quality reduction hidden as optimization |
| P2 | Research claims exceed demonstrated modules if presented as the entire original 12-stage stack | README, `ARCHITECTURE.md`, `docs/METHODOLOGY.md` | Label candidates/extensions correctly; audit actual invocation, not import availability, for MASt3R/COLMAP, SAM 2 video memory, TSDF fusion, grounded explanations and navigation |

All module names in the table are under `src/spatialmemory/` unless otherwise noted.
This is the known failure backlog, not a claim that every repository line has
already been independently reviewed. Inspect the implementation before assuming
a named module is absent or complete.

## Critical diagnostic corrections

- The white pump bottle was behind the pink handled container in BEFORE. Seeing
  it in the AFTER crop may be exposed background; **this alone does not prove
  the crop is misplaced**. Earlier statements asserting misplacement were too
  strong and have been corrected in the audit.
- High point confidence previously bypassed generating-pixel reprojection checks.
  That defect is fixed. For the source observation, accepted points decreased
  from 2,692 to 708 under the 2-native-pixel limit. This does not solve depth
  accuracy or cross-view identity.
- Depth backprojection fits its own generating ray by construction. It cannot
  serve as independent validation of that depth estimate.
- Point-map and depth-head agreement from the same neural network is correlated
  evidence, not independent ground truth.
- Fixed tolerance values or raw model confidence are not calibrated uncertainty.
- Do not convert all UNKNOWN decisions into REMOVED or hardcode the known object.

## Failed experiments: do not repeat unchanged

- Generic crop-only detection mislabeled the container. Four rotations did not
  repair its identity/label, although they recovered a neighboring bottle.
- Background-only local planar alignment rejected all 45 tested AFTER views for
  the tested source observation. It is a diagnostic, not a successful fallback.
- SmolVLM-500M structured run: 130.97 sec, returned only a frame heading.
- SmolVLM ~2B structured run: 359.05 sec with mixed attention, invented frame IDs
  and an invalid ADDED entry. Zero accepted proposals. More parameters alone did
  not fix the task.
- All-eager larger-model test exceeded ten minutes without output and was stopped.
- All-SDPA failed: the installed Idefics3 vision encoder does not support it.
  Tested compatible combination: eager vision encoder, SDPA text decoder.
- A full run without code/configuration changes is regression validation, not a
  reason to expect better detection. Use cached diagnostic replay to isolate fixes,
  then fresh inference for an acceptance test.

## Private files available in the current workspace

BEFORE (do not reverse roles):
`/workspace/attachments/a5ac41bf-942c-4fce-bf9d-f376005009cc/before room.mp4`

AFTER:
`/workspace/attachments/17db9d20-db27-4ef4-b195-933f2a812511/after  room.mp4`

Fresh run:
`/workspace/spatialmemory-fresh-validation-20261015-r2/`

Configuration/logs/tests:
`/workspace/final-validation-20261015/`

Dense scene:
`/workspace/final-dense-scene-20261015/`

Verified validation ZIP:
`/workspace/spatialmemory-validation-20261015.zip`

Cached diagnostic run:
`/workspace/spatialmemory-localization-fix-20261014/`

Ray-level diagnostic:
`/workspace/removal-gate-audit-20261014/result.json`

Larger VLM actual output:
`/workspace/structured-visual-2b-mixed-20261014/visual_changes.json`

Models and environment:
`/workspace/spatialmemory-models/`, `/workspace/spatialmemory-venv/`,
`/workspace/spatialmemory-thirdparty/`.

These are actual paths from this workspace, not paths guaranteed on a new
machine. Locate/transfer files before invoking commands. Keep private scene data,
licensed datasets and checkpoints out of Git. Older mug/perfume recordings and
other datasets must not be confused with this bathroom/cabinet pair.

## Reproducible commands

```bash
cd /workspace/Spatial-AI-system-for-Humanoid-Robotics
OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 /workspace/spatialmemory-venv/bin/python -m pytest -q

# Fresh inference: output directory must be new; existing config disables reuse.
OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 /workspace/spatialmemory-venv/bin/spatialmemory \
  '/workspace/attachments/a5ac41bf-942c-4fce-bf9d-f376005009cc/before room.mp4' \
  '/workspace/attachments/17db9d20-db27-4ef4-b195-933f2a812511/after  room.mp4' \
  --config /workspace/final-validation-20261015/config.json \
  --output /workspace/claude-fresh-validation

# This export reuses the computed neural maps, not fresh model inference.
OMP_NUM_THREADS=4 /workspace/spatialmemory-venv/bin/python -m spatialmemory.dense_export \
  --run /workspace/claude-fresh-validation --output /workspace/claude-dense-scene
```

For a cached replay, start from
`/workspace/localization-audit-20261014/config.json` and inspect its reuse paths.
Do not overwrite outputs or silently change inference settings while calling it
a comparable benchmark. Record dependency versions, seeds, model revisions,
input hashes, frame IDs, transforms, runtime and acceptance decisions.

## Working sequence and definition of done

1. Confirm clean/dirty Git state and available private artifacts; preserve user work.
2. Reproduce the specific ray/camera discrepancy before proposing a root cause.
3. Fix one demonstrated problem; exercise targeted tests and saved real evidence.
4. Obtain an **automatic**, image-supported visual change proposal without answer
   key injection. A manual review or valid JSON alone does not meet this milestone.
5. Validate that proposal's location in the shared 3D scene with actual geometry
   and visibility evidence; preserve abstention where those checks fail.
6. Run a fresh end-to-end inference, save before/after crops and an interactive
   3D output, and inspect them. One successful pair demonstrates feasibility only.
7. Test independent added, moved, removed, unchanged, occluded and lookalike cases;
   report valid per-class metrics, failures and ablations. Do not tune on held-out
   evaluation examples or invent annotations.
8. Test the actual deployed UI and downloadable artifact. Document deployment,
   resource requirements and unresolved limitations. Push only code and permitted
   reports, then report the commit hash and exact tested outcomes.

The user needs working automatic results, not more manual labels presented as a
model improvement. If a stage fails, identify it precisely and keep its raw
output; do not claim the system is complete or perfect.
