# Methodology and implementation status

The research method is **Visibility-Aware Multi-View Probabilistic Consensus for 4D Semantic Change Detection**. Its architectural candidates do not imply that every proposed model is installed or validated.

## Twelve-stage review

| Stage | Implemented behavior | Remaining validation or work |
|---|---|---|
| Multimodal input | Two RGB recordings; typed optional calibration, depth, poses and IMU | General LiDAR/IMU fusion is incomplete |
| Frame intelligence | Quality/diversity selection; original frames retained as evidence source | Real capture coverage validation |
| Neural reconstruction | Actual pinned VGGT inference and source/checkpoint checks | Monocular scale and reconstruction accuracy |
| Geometric cross-validation | Sparse geometry and explicit sensor-assisted path | Full MASt3R/COLMAP and bundle-adjustment integration |
| Shared registration | Foreground-excluded Sim(3) RANSAC; phone profile requires at least four held-out correspondences; sourced RGB-D transform with gated ICP | Autonomous RGB-only cross-session reliability and calibrated covariance |
| Semantic perception | Grounding DINO, SAM 2 image masks, DINOv2 descriptors | Annotated detector/segmentation quality; YOLO is not part of the default stack |
| Temporal identity | Optical-flow diagnostics and multi-cue one-to-one assignment with unmatched options | Robust cross-session identity and full SAM 2 video-memory tracking |
| 4D world model | Persistent observations, histories, snapshots and limited spatial relationships | Full dynamic occupancy/orientation/uncertainty propagation |
| Evidence fusion | Conservative gates and optional supervised likelihood fitting | Independent held-out deployment calibration |
| Occlusion-aware changes | Predicted/measured visibility provenance, free-space checks and bounded movement | Validated error budgets and real confirmed changes |
| Semantic/navigation intelligence | Computed grounded-to-evidence summaries; standalone experimental A* | Grounded VLM and validated floor-map/robot integration |
| Validation/visualization | Actual clouds, evidence, JSON/PLY/ZIP, local UI and temporary Colab wrapper | Genuine phone benchmark, independent annotations and generalization |

## Decision integrity

An invalid registration stops the pipeline. Small correspondence sets may pass the baseline fitting gate but do not provide held-out validation; the phone configuration now rejects them. Held-out correspondence residuals are not independent scene-level accuracy or covariance.

Monocular distances are reconstruction units unless scale is explicitly verified. Predicted depth is not measured depth. Affinity scores, detector scores and predicted mask IoUs are not measured correctness probabilities. UNKNOWN is an abstention rather than verified unchanged status.

Association requires semantic compatibility, valid descriptors and localization, bounded search, adequate independent viewpoints and nonambiguous assignment. Descriptor normalization rescales before computing the norm to avoid overflow while preserving direction. Bounded movement requires explicit validated error budgets; ICP RMSE alone is not such a budget.

Known changes are reserved for evaluation. Source-derived partial labels, synthetic transformations and independently reviewed real observations remain distinct. No real annotation is invented or inferred from absence in a single frame.

## Evidence and reproducibility

[Application verification](../reports/APPLICATION_VERIFICATION.md) documents actual neural execution on a dataset RGB control. [3RScan validation](3RSCAN_VALIDATION.md) documents approved source-assisted alignment and the missed single supported MOVED instance. [Repository review](../reports/repository_review.json) documents current regression results and cleanup. Historical run reports remain available and retain their original scope.

The animated user-uploaded example produced 25 UNKNOWN decisions. Its 16/17-inlier alignment had zero held-out correspondences; this is not a genuine phone benchmark and does not satisfy the updated phone gate. Earlier outputs are preserved rather than rewritten as if the stricter gate had applied retrospectively.

The Colab UI invokes the same core pipeline on CPU. Local wrapper/browser checks do not establish successful installation in a fresh Google runtime or public-relay availability. No training was restarted and no architecture/checkpoint was replaced.

Model and dataset licences apply independently of project-code licensing. Commercial eligibility, production deployment and robot safety are not established.

## Small-object evidence repair

### Portrait input coverage

VGGT's upstream default `crop` preprocessing center-crops tall inputs after
resizing the width to 518 pixels. On a 464×832 recording it retains approximately
original rows 183–649. Objects near the lower edge can be clipped even though they
are visible in the original video. The project defaults to
`neural_preprocessing: "pad"`, which preserves
the full image using the upstream padded mode. Feature extraction and predicted
camera parameters use the same processed image. Padding pixels are excluded from
keypoint geometry, confidence quantiles, object localization and visibility support.
Cached geometry cannot be reused across preprocessing modes. Padding reduces
effective object resolution and is not a guarantee
of improved reconstruction or labels.

### Every-frame temporal diagnostics

Temporal neural re-detection uses the original decoded frame resolution. Its
masks are mapped with nearest-neighbor interpolation to the reduced optical-flow
resolution for association. Evidence crops retain original pixels, and each
re-detection records both perception and tracking dimensions. A detection whose
mask disappears at flow resolution cannot establish a temporal identity link.
This preserves neural input detail without treating 2D tracking as 3D change proof.

Set `temporal_tracking_enabled` to true with semantic inference enabled to decode
both complete videos again, preserving ingestion timestamps. Neural keyframe
masks initialize independent 2D segments; forward/backward optical flow propagates
them between anchors. Low flow support or a mask-area ratio outside 0.5–2 stops
propagation and records `requires_redetection`. A real Grounding DINO / SAM 2 / DINOv2 callback re-detects periodically
and after drift (with a 0.5-second cooldown). Drift requests remain pending
through the cooldown. Supported live tracks can link to new detections only
with matching normalized labels, appearance cosine at least 0.85, mask IoU at
least 0.2, and a 0.05 ranking margin against competing links on either side.
Lost or ambiguous tracks begin independent segments. These are uncalibrated
2D identity proposals; no occlusion or removal conclusion follows from flow loss. JSONL diagnostics record every decoded frame and neural provenance. Optical flow
is computed once per frame pair and shared across object masks. This optional CPU stage
can be expensive and does not replace or add independent 3D consensus evidence.

Repeated detector words and the clear cup/mug synonym are normalized for
association. Mixed category phrases remain ambiguous. Before segmentation,
same-category proposals overlapping at IoU 0.8 or greater are suppressed in
score order before applying the detection budget; nearby separate instances
and conflicting categories remain separate. Raw detector outputs and selected
indices are retained for audit. This improves proposal handling across rooms,
but does not establish object identity or repair unsupported depth. A replay of
the previous phone pair's detector outputs suppressed zero selected proposals;
no improved removal accuracy is claimed from this change.

### Strict visibility-state API

`spatialmemory.object_state.evaluate_object_state` separates current-view PRESENT,
HIDDEN, REMOVED and OUT_OF_VIEW states from cross-session change labels. It
requires verified whole-object geometry, identity and position for presence,
complete occlusion for hidden state, and at least two independent free-space
views plus a verified prior object for removal. Missing or contradictory evidence
raises `ValidationFailureError`; detector absence alone never establishes removal.
Its confidence remains null because a validated state-specific calibrator is not
available. This API does not certify existing predicted depth or automatically
convert existing UNKNOWN change events into verified states. Pipeline events now
also expose explicit validation failures and null calibrated state confidence.

The generic indoor detector vocabulary now covers cups, mugs, bowls, bags and toiletries. The phone/app profile permits up to sixteen detections per frame at the existing score threshold. Compound labels remain model proposals; their instance identity is not verified.

VGGT depth-head confidence is retained. Pixels rejected by the point-head median may contribute only when depth-head confidence passes its own frame median, depth is positive/finite, and depth backprojection through `T_camera_world` agrees with the world point map within the configured voxel tolerance. Localization and visibility share this support policy. Legacy surfaces without depth confidence retain their original filter.

The heads share a model and training; this is a consistency check, not independent geometric validation, metric calibration or a calibrated uncertainty bound. Identity and viewpoint gates remain unchanged. Known-pair development still fails to verify the user-reported removals; an independent benchmark is required. See [review and tests](../reports/object_evidence_repair.json).
