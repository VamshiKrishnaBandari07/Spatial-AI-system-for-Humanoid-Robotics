# References and related work

These papers describe upstream methods used by the perception components.
Their published results are not benchmark results for SpatialMemory-X.
Checkpoint versions, licences and actual invocation are documented separately
in the model configuration and run provenance.

| Reference | Role in this project | Boundary |
|---|---|---|
| [Grounding DINO: Marrying DINO with Grounded Pre-Training for Open-Set Object Detection](https://arxiv.org/abs/2303.05499) | Text-prompted object bounding-box proposals, including native crop refinement | A proposed class or raw detection score does not verify object identity or physical change. |
| [SAM 2: Segment Anything in Images and Videos](https://openreview.net/forum?id=Ha6RTeWMd0) | Object segmentation masks used for crops, appearance features and geometry filtering | Mask quality remains model-dependent; citing video segmentation does not establish that every run uses the SAM 2 video predictor. |
| [DINOv2: Learning Robust Visual Features without Supervision](https://arxiv.org/abs/2304.07193) | Normalized visual appearance embeddings for candidate association | Similar embeddings can represent lookalikes; geometry, temporal evidence and ambiguity checks are still required. |
| [Sigmoid Loss for Language-Image Pre-Training (SigLIP)](https://arxiv.org/abs/2303.15343) | Image/text similarity checks over a fixed object-label vocabulary | Similarity ranks and margins are not calibrated detection or change probabilities. |

The system combines these upstream components with shared-scene registration,
object-centric tracking and explicit multi-view visibility evidence. The
research direction is **Visibility-Aware Multi-View Probabilistic Consensus
for 4D Semantic Change Detection**. Reliable smartphone change detection and
deployment calibration remain unresolved; this title states the research
direction, not a validated novelty or accuracy claim.

See [methodology](METHODOLOGY.md), [refinement limits](MULTIVIEW_REFINEMENT.md)
and [measured correspondence experiments](CORRESPONDENCE_EXPERIMENTS.md).
