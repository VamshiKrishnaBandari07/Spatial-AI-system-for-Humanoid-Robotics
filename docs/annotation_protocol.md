# Ground-truth annotation protocol

Ground truth comes from documented observations and independent review, never directly from model predictions. Schema validity and reviewer names are provenance declarations, not authentication or proof of label correctness.

## Establish the scene pair

Use official reference/rescan metadata or independently verified capture logs identifying the same physical location and both distinct visits. Retain original metadata, source URL and checksum. Confirm static overlap and coordinate conventions before geometric evaluation. Geometrically similar rooms and two unrelated scene IDs are not pairing evidence. A copy of one recording is a loader control, not a changed scene.

`PairEvidence` accepts official 3RScan relationship arrays or `manual_capture` JSON with `before_recording_id`, `after_recording_id`, `same_environment_verified`, `capture_location_evidence`, `verified_by` and documented physical `documented_changes`. These fields must describe actual evidence. The manual format does not supply external verification by itself. The `unverified` format always fails the scene relationship gate.

## Annotation levels

- **Level 1:** persistent object ID, semantic label, change class, visibility and provenance. No coordinates are required.
- **Level 2:** add frame-linked 2D boxes or mask references; identify the recording and original frame. Boxes use nonnegative pixel xyxy coordinates and require positive area.
- **Level 3:** add actual 3D position observations, coordinate frame, units and scale status. A 2D mask is not mandatory at this level.
- **Level 4:** add full geometry asset references and verified correspondence; MOVED records include measured displacement. Geometry files remain outside Git. These references do not automatically enable IoU evaluation.

Supported core categories: ADDED, REMOVED, MOVED, UNCHANGED, UNKNOWN. ROTATED/DEFORMED are not implemented as independent classification targets in the current core; retain original dataset labels in source provenance rather than silently coercing them. Use UNKNOWN for unsupported or ambiguous cases.

ADDED cannot have an observed before object; REMOVED cannot have an observed after object. Missing observations alone do not establish addition or removal. A reviewer must inspect the relevant location with sufficient visibility, preserving occlusion/partial-coverage ambiguity. MOVED requires correct identity association and actual displacement evidence. Do not infer metric coordinates from unscaled monocular predictions.

## Human workflow

Open `/annotate` in the local application. Choose both recordings, navigate independently to corresponding views, name the scene and each persistent object, provide native before/after IDs where available, record a class and visibility, and describe the source evidence. The page captures current playback times for new observations; equal visit timestamps do not establish matching viewpoints. Original videos stay in the browser and are not uploaded by this page. If the browser cannot play the codec, create a distinct H.264 MP4 preview copy without overwriting the original; retain the original source identity and timestamp relationship.

The interface exports Level 1 annotations and can import/review richer schema-valid annotations while preserving their existing observation/geometry fields. Unsaved annotations are page-local; export before closing. Full 3D annotation authoring and automatic viewpoint matching are not provided.

A second reviewer must independently inspect the original evidence. Supply distinct annotator/reviewer identifiers, timezone-bearing review timestamp and source provenance. A model suggestion can be marked `model_suggestion_verified` only after source-based human verification; unverified `model_generated` records cannot enter a benchmark. Preserve disagreement as UNKNOWN or unreviewed. Review of the current prediction in the comparison dashboard is a separate human assessment, not independent ground truth.

`POST /annotations/validate` validates schema only and always returns `benchmark_ready: false`. Imported records are not automatically certified. For benchmark gates, set `data_origin: REAL` only for actual real captures, retain top-level `provenance`, and independently review every included annotation. `exhaustive_object_inventory: true` requires a complete inventory within the explicitly reviewed scope; leave it false for partial labels.

## Evaluation and identity association

Generate predictions without annotations. Independently associate external ground-truth IDs with local prediction IDs in an `IdentityMapping` JSON (`scene_id`, `ground_truth_to_prediction`, `provenance`, `verified_by`). Association must be one-to-one; do not silently match by label, predicted class or nearest position. Missing associations remain UNKNOWN. Extra predictions are retained in the audit. UNKNOWN ground truth is excluded from known-class scoring and UNKNOWN predictions count as abstentions and incorrect known-class predictions.

`python scripts/benchmark_scene.py --manifest MANIFEST --pair-evidence PAIR_EVIDENCE --annotations ANNOTATIONS --report REPORT --identity-mapping MAPPING --project-root ROOT --output NEW_REPORT_DIRECTORY` validates source checksums, same-scene relationship, documented changes, independently reviewed real annotations, input hashes and registration before metrics. A report cannot be substituted from different recordings. Failed gates yield a pair/failure report, not fabricated metrics. Completed metric output is not overwritten.

Classification precision/recall/F1/confusion are scoped to known annotated identities. Exhaustive inventories additionally enable physical-change precision/recall/F1 including extra detections, verified-negative false-positive rate, and incorrect-removal proposal fraction. False-removal fraction is not the same denominator as false-positive rate. Without an exhaustive inventory those quantities stay unavailable. Localization, box/region IoU, calibration, and uncertainty coverage are not yet implemented and must not be inferred from classification scores.

Object matching accuracy requires separate source-based before/after correspondence truth and explicit predicted associations; the evaluation identity mapping itself is not a matching prediction. Peak memory is unavailable from old reports. `scripts/profile_inference.py` records actual wall time and maximum child RSS in a fresh inference process; this is not aggregate process-tree or GPU memory. Supply `--resource-profile` to benchmark_scene only for a matching report hash. Keep RGB-only, RGB-D, synthetic and real results separate. Current reconstruction profiles are RGB-only even when optional sensor metadata is retained.
