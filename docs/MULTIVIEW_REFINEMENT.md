# Multi-view object refinement

Use `configs/phone_multiview.json` or the UI's **Multi-view object analysis**
option for the experimental small-object workflow. It performs real neural
full-frame/tile detection, up to four candidate-centred crop passes per selected
frame, SAM segmentation, DINOv2 appearance extraction and SigLIP label checks.
Crop passes use image evidence, not answer-key object locations. Raw scores,
alternate proposals and crop rectangles remain available for diagnostics.

## Geometry

Native pixel mapping, right-handed camera transforms, parallax and positive
depth checks precede acceptance. Triangulated keypoints are optimized with
SciPy Huber-loss least squares before the final gate. The reusable
`reprojection.refine_point` accepts V views, known camera matrices and optional
pixel uncertainty. Uncertainty weights optimization but does not relax the
strict final **less than 2 px in every supplied view** acceptance criterion.
The current correspondence recovery calls it on two-view tracks; this is
point-only bundle adjustment, not joint camera/intrinsics optimization.
After robust fitting, a symmetric least-squares polish is compared using the
worst unweighted pixel residual. It is retained only when that residual is
lower and all depths are positive. On one saved mug pair, the median of the
per-point worst-view error fell from 9.07 to 5.25 px, still above the 2 px gate.
The cached-evidence rerun confirmed zero changes; this is an optimization
improvement, not successful change detection.

Optimizing an incorrect correspondence cannot guarantee acceptable geometry.
Object centroids are summaries of accepted surface points: a centroid is not
a surface keypoint and has no corresponding observed pixel unless separately
annotated. Consequently no centroid reprojection accuracy is claimed.

Camera validation uses static-scene SfM, reference correspondences, RANSAC and
held-out checks. A lost adjacent pose can reacquire against the last registered
frame. Rejected camera validation is recorded; it does not force alignment.
Monocular reconstruction scale and absolute calibration remain unverified.

## Calibration

`DetectorTemperature.fit(logits, correctness_labels, source, model_revision)`
fits positive scalar temperature by binary negative log likelihood. Supply
both correct and incorrect detection labels from validation scenes. Keep
independent test scenes out of fitting. Serialize the fitted dataclass with
`dataclasses.asdict` as JSON and set `detector_temperature_file` to that path.
The detector checks the revision and preserves actual retained query logits,
raw scores and fitted detection probabilities in observation provenance.
No fitted file is distributed because verified calibration labels are absent.

These probabilities describe detection correctness, not physical removal.
They are exposed for diagnostics; final change consensus continues to use
its separate labelled evidence likelihood calibration when supplied. Detector
temperature alone never authorizes a change or a deployment-calibration claim.

## Identity and absence

Set `observed_surface_visibility_enabled: true` to test visibility using up to
32 actual computed object-surface points retained in observation provenance.
Samples are transformed with the observation into the shared frame; cache
reuse recomputes them from current geometry. This avoids treating box corners
on furniture or in empty space as observed object surfaces. Missing samples
fail closed rather than falling back to a box. This option is experimental
and does not relax depth support, distinct-view or patch-validation gates.

The real cached-evidence test still confirmed no changes. For the mug's 32
samples across six comparison views, 164 rays failed depth support and 28 were
free; none were outside the image. The new ray diagnostics distinguish invalid
depth, rejected depth support, out-of-image and behind-camera conditions.
This isolates the remaining visibility failure without claiming model-depth
accuracy or physical absence.

The joint matcher combines normalized geometry, appearance and class costs,
gated Hungarian assignment, competitor ambiguity rejection and EMA galleries
with a temporal memory buffer. Nearby lookalikes retain strict appearance
requirements. Native trajectory corroboration is not an across-session
identity shortcut.

Advanced removal evidence requires at least two distinct free-space views,
plus cross-view textured background/depth patch verification. Missing
detections, depth holes and untextured patches cannot establish removal.
Predicted-depth agreement is evidence, not mathematical proof of physical
emptiness or a calibrated sensor uncertainty bound.

## Time and diagnostics

Manifest recordings optionally accept timezone-aware ISO-8601 `recorded_at`.
Supply it for both roles or neither. Reversed/equal chronology is rejected;
filenames never establish recording order. Without capture timestamps,
explicit Before/After roles remain authoritative. Frame ingestion retains
decoded presentation timestamps, including variable frame rate.

Observation provenance carries crop/mask paths, raw and calibrated detection
scores (when fitted), appearance features and per-pair reprojection diagnostics.
Visibility evidence carries sampled occupancy states, supported fractions and
peer frame IDs. These are available to UI/API consumers; no new dedicated
diagnostic panel is claimed.

## Validation limits

Regression tests validate code behavior, not generalization. Earlier real
phone-pair inference confirmed zero changes. The updated code completed a
fresh neural perception run with verified cached reconstruction in 579.99 s:
107 inconclusive decisions and zero confirmed changes. Geometry recovered six
before and eight after object localizations, but identity and positive
free-space support remain insufficient. Independent annotated benchmarks
are still required; no precision, recall or F1 is claimed. Raw videos, crops, model weights
and restricted datasets must remain outside Git.
