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
