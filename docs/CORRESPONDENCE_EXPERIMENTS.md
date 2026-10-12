# Camera and learned correspondence experiments

These optional candidates do not replace the default model stack or relax
geometry/identity/visibility gates. Real saved phone evidence was used for
the experiments below. Cached detector/SAM/DINOv2/SigLIP outputs were explicitly
reused; LoFTR itself was loaded and invoked locally in its experiment.

## Joint camera adjustment

Enable `classical_bundle_adjustment_enabled` together with classical camera
validation to optimize camera rotation/translation and landmark positions.
Intrinsics and the first two cameras are fixed to retain the scale gauge.
Twenty percent of each camera's track measurements are withheld. Training
improvement, positive depths, bounded rotation changes, held-out median and
95th-percentile errors, and held-out 2 px inlier counts determine acceptance.
Rejected candidates retain original poses and record the rejection.

On the real pair, held-out median error changed from 0.453 to 0.488 px before,
and 0.612 to 0.637 px after. Both candidate refinements were rejected.
This trial did not improve small-object localization or detect changes.

## Learned small-crop matching

Install `requirements/neural.lock.txt`. Obtain the optional checkpoint using
the public Kornia model repository with its immutable revision:

```python
from huggingface_hub import hf_hub_download
checkpoint = hf_hub_download(
    "kornia/loftr", "loftr_indoor_ds_new.ckpt",
    revision="0fb3b456cf176746709c38fa9cf80695a1c82dac",
)
```

Set `learned_matching_checkpoint` to that local path. The SHA-256 must be
`be9ff88b323ec27889114719f668ae41aff7034b56a4c4acbd46b8b180b87ed3`.
The module loads with `weights_only=True`, strict state-dict checks and no
remote Python execution. The public Kornia model card declares MIT; original
LoFTR source declares Apache-2.0. Review both before distribution/use.
Checkpoints and scene images remain outside Git.

Crop proposals derive from actual object masks, retain context and map learned
correspondences back using the resize pixel-centre affine. Only native endpoints
inside both eroded masks and raw neural scores >=0.8 are considered. These
scores are not calibrated probabilities. Small candidates can use up to eight
learned pair attempts per recording (`learned_matching_max_pairs`).
Learned and classical point proposals are evaluated separately against the
same cameras. A learned proposal must improve accepted support and retain at
least 50% geometrically valid points; downstream ambiguity, positive-depth,
parallax and strict <2 px rules remain unchanged.

Actual inference produced 3 masked matches before and 18 after, but no accepted
learned triangulated points. The mug's tested pairs produced only one and zero
masked learned matches, below the four-point support requirement. The run
completed in 43.45 seconds with 107 inconclusive decisions and zero confirmed
changes. This rules out claiming that adding LoFTR solved this pair.

## Delivery boundary

The project exports shared point clouds, an interactive viewer, object crops,
decision explanations and diagnostic failures. Reliable added/removed/moved
classification on these phone videos remains unproven. Tests demonstrate
implementation behavior, not real-world accuracy. No F1 or precision is
reported without defensible independent change annotations.
