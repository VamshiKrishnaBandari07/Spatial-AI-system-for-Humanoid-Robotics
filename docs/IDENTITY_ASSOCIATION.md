# Joint 3D object identity association

Use `configs/phone_identity.json` to enable the new `joint` association mode.
Existing association modes and UI defaults remain available and unchanged.

```bash
python -m spatialmemory_x run --before before.mp4 --after after.mp4 \
  --scene-id room_01 --config configs/phone_identity.json --output results/room_01
```

The module `spatialmemory.identity` exposes typed `IdentityObservation`,
`IdentitySettings`, `AssociationResult` and `IdentityGallery` APIs. Inputs require
normalized ReID embeddings and centroids in a common registered coordinate frame.
Geometry scales are expressed in that frame's units; they are not automatically
metres. AABB extents must have positive size. Missing geometry remains unresolved.

For N tracks and M observations, the returned cost matrix has shape N×M.
Every eligible component lies in [0,1]:

- Centroid distance: `min(norm(x-y)/radius, 1)`.
- Box distance: `1 - IoU3D`, blended with centroid distance when both boxes exist.
- Appearance distance: `(1 - cosine(normalized_embeddings))/2`.
- Class distance: zero for normalized compatible labels, otherwise one.

The weighted sum is divided by the sum of weights. Scores are ranking costs,
not calibrated identity probabilities. A class mismatch incurs a soft cost but
also requires appearance cosine at least 0.98. The appearance floor rises by
`proximity_boost * max(0, 1 - distance/proximity)` for nearby candidate positions.
Spatial radius is a hard search bound; larger movements require an appropriate
radius and stronger competing-candidate checks, not a forced association.

The matcher rejects row and column ambiguities before assignment, including
nearby lookalikes with indistinguishable appearance. Hungarian assignment uses
N+M dummy rows/columns to permit unmatched objects. Ambiguous observations do
not update EMA prototypes. The result retains gate reasons and raw component
measurements for every candidate.

## Temporal memory and integration

`IdentityGallery.update` consumes one timestamped frame/segment batch. Each
accepted link updates the normalized embedding with EMA. Occluded tracks remain
eligible for `identity_memory_batches` missing batches. This is a batch count,
not a time duration. Session changes can restart timestamps, but cannot change
the shared coordinate frame. Scene-specific namespaces prevent IDs from being
confused between separate galleries. Gallery updates are sequential, not thread
safe. Check `gallery.tracks[id].confirmed`: returned IDs remain provisional until
`identity_confirmation_hits` supported sightings. Repeated observation IDs cannot
manufacture confirmation. Ambiguous evidence is audited; it is not retrospectively
assigned an identity without new supporting observations.

```python
from pathlib import Path
from spatialmemory.identity import IdentityGallery, IdentitySettings

gallery = IdentityGallery(IdentitySettings(), "room_shared_world", "room_01")
assignments = gallery.update(observations, timestamp=1.0,
                             session="recording_before",
                             coordinate_frame="room_shared_world")
gallery.save(Path("results/room_01/gallery.json"))
# A later recording must already be registered into the same frame and units.
gallery = IdentityGallery.load(Path("results/room_01/gallery.json"))
```

In the core pipeline, per-recording galleries group real 3D semantic observations;
their EMA descriptors, linked observation histories and geometry feed directly
into cross-recording joint matching and consensus. Central consensus commits
shared IDs only after its identity and independent-view checks. The separate
full-video 2D flow diagnostic is not promoted to 3D identity evidence.

Outputs include `diagnostics/{before,after}_identity_checkpoint.json`, reloadable
local gallery memory, and `diagnostics/{before,after}_identity_gallery.json`,
per-observation support audits. Existing `diagnostics/association.json` records
cross-recording candidate gates. Checkpoints contain appearance embeddings and
observation metadata; keep them with private run artifacts, outside Git.

## Validation and limits

Synthetic tests exercise costs, lookalike rejection, unmatched objects, dynamic
thresholds, class confusion, EMA, occlusion, expiry, session handoffs, checkpoint
restoration and core grouping/consensus integration. These do not measure real
identity accuracy. A saved-observation replay from the supplied phone recordings
produced three confirmed local tracks but no accepted cross-recording identities;
independent-view support remained insufficient. This replay did not rerun neural
models or establish object-removal accuracy.

The module cannot repair inaccurate depth, missing detections or incorrect
registration. Observed boxes are partial AABBs, not measured full object volumes.
Identical nearby objects can remain unresolvable. Confidence calibration and
independent annotated benchmarks are still needed before deployment claims.
