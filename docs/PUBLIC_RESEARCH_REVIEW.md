# Public implementation review — 2026-10-13

This review inspected official README files, selected implementation files and
licenses. No external model was benchmarked here and no improvement in detection
accuracy is claimed. Downloads were limited to source text, not datasets or weights.

## Findings tied to our failures

### OpenMask3D: object-specific views and crops

Sources: [repository](https://github.com/OpenMask3D/openmask3d),
[feature implementation](https://github.com/OpenMask3D/openmask3d/blob/main/openmask3d/mask_features_computation/features_extractor.py),
[license](https://github.com/OpenMask3D/openmask3d/blob/main/LICENSE).

The inspected implementation projects 3D points into camera views, compares their
depth to sensor depth, groups visible pixels by instance mask and ranks top-k
views per mask. It derives object-specific bounding crops and aggregates image
features. This directly addresses our whole-frame matching failure.

Important adaptation: existing-object depth agreement must NOT be required to
retrieve a view of an empty former location. Rank geometric coverage/resolution
first, then independently evaluate FREE/OCCUPIED/OCCLUDED/INSUFFICIENT using the
ray-box module. Otherwise selection excludes the very absence evidence needed.
Its sensor-depth tolerance is not a calibrated bound for our monocular predictions.

The documented inputs include a scene point cloud and posed RGB-D frames. The
repository uses an older Python/CUDA-oriented stack and separate mask/SAM/CLIP
checkpoints. Its MIT source license does not automatically license all dependencies,
weights, or datasets. Adapt concepts into our code rather than replace the environment.

### ConceptGraphs: persistent object maps

Sources: [repository](https://github.com/concept-graphs/concept-graphs),
[mapping code](https://github.com/concept-graphs/concept-graphs/blob/main/conceptgraph/slam/mapping.py),
[license](https://github.com/concept-graphs/concept-graphs/blob/main/LICENSE).

The inspected code computes detection-to-map spatial overlap/IoU and cosine image
similarity, combines them, then merges detections into stored objects. This is a
useful object-map architecture for aggregating observations and reducing fragments.
Its greedy per-detection argmax merge is not our requested one-to-one cross-session
matching and does not resolve ambiguous lookalikes. Preserve our gated assignment,
record alternatives, and evaluate track merging separately. Source license: MIT.
This is not a drop-in added/removed/moved benchmark system.

### CUT3R and DUSt3R: reconstruction alternatives, not removal detectors

Sources: [CUT3R](https://github.com/CUT3R/CUT3R),
[CUT3R license](https://github.com/CUT3R/CUT3R/blob/main/LICENSE),
[DUSt3R](https://github.com/naver/dust3r),
[DUSt3R license](https://github.com/naver/dust3r/blob/main/LICENSE).

CUT3R offers continuous perception with persistent state, worth a controlled pose/
depth consistency comparison when adequate compute is available. Its documented
setup uses CUDA, and the README notes memory growth with encoded frame count.
DUSt3R is a reconstruction alternative. Neither inspected project establishes
visibility-aware removal decisions for our videos. Both inspected licenses are
CC BY-NC-SA 4.0; they are not unrestricted commercial deployment choices.
Do not replace VGGT solely because a different model is newer or larger.

## Dataset selection

[3RScan](https://github.com/WaldJohannaU/3RScan): documented related scans, RGB-D,
poses, instance IDs, reference/rescan transforms and rigid/nonrigid object metadata.
Use approved existing files first. Audit actual per-pair annotations; missing IDs
are not automatically removal labels, and rigid transforms do not provide a full
addition/removal answer key. Scene-family splits must keep related rescans together.
Access remains subject to official terms, regardless of public source code.

[ARKitScenes](https://github.com/apple/ARKitScenes): smartphone/tablet RGB-D, poses,
reconstruction and depth-upsampling tasks. Use suitable reference-depth assets for
held-out error evaluation, retaining synchronization and sensor/reference uncertainty.
Do not label ordinary sequences as changed scenes. Dataset-specific errors cannot
be assumed exchangeable with arbitrary phone RGB predictions. Recheck the dataset
license before redistribution or commercial use.

## Concrete implementation/evaluation order

1. Build a persistent object support map from native masks, descriptors and real
   reconstruction points, retaining provenance and unresolved competing identities.
2. Project each object's support/volume into AFTER cameras. Retrieve views by
   expected-region coverage and useful native resolution, independent of occupancy.
   Do not require a global planar homography across the entire room.
3. Run object crops through existing detection/segmentation; compare identities
   jointly across recordings. Search outside the original location for movement.
4. Apply ray-box depth intervals only with explicit applicable error bounds.
   Obtain depth residual calibration from held-out reference-depth examples;
   report uncertainty-model transfer limits. Reference/pose errors also contribute.
5. Produce two evidence levels: visual change proposal with frames; geometrically
   supported 3D event only when localization, identity and visibility pass. Do not
   give a visually obvious change invented 3D coordinates or calibrated confidence.
6. Evaluate reconstruction, object matching and physical-change decisions
   separately. Include unchanged, occluded, moved lookalike and removed examples.

## Current outcome

The review supports object-centric view retrieval as the next implementation
priority. It does not prove that another checkpoint will fix our system. The
latest full phone run still has zero confirmed changes. The recent ray-box adapter
is diagnostic-only because applicable uncertainty bounds are absent. The local
500M VLM failed this pair. No new accuracy or F1 score is available.
