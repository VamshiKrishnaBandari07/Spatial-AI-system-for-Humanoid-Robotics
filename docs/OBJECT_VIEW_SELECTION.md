# Expected-object view selection

`object_view_selection.py` projects target-world object support points into each
camera using its actual intrinsics and pose. It measures the fraction inside the
unpadded image content and projected pixel area. Rankings use coverage first,
resolution second. Behind-camera and padding samples count against coverage.

`semantics.visibility_evidence` now includes per-view `object_projection` and
`ranked_object_view_ids`. Existing registration inversion happens before projection.
Ranking does not depend on observed depth agreement, so an empty expected location
is not excluded merely because the original object's surface no longer exists.
This ranking is available to downstream crop selection; it does not by itself run
additional detections or replace cross-recording identity matching.

A diagnostic replay of stored original VGGT camera predictions and before-object
boxes found candidate after views for 27/27 tracks with extents. This is neither
27 unique verified objects nor 27 changes. Partial boxes, erroneous object geometry,
pose uncertainty and occlusion can all make a projected region misleading. The
projection grid is the reconstruction grid, not native RGB; crop consumers must
use the existing pixel mapping to convert coordinates. Full object visibility,
metric scale and occupancy are not certified by this selector.

Validation: 319 regression tests pass, including exclusion of wrong-camera views,
padding, and behind-camera support. No new full-video inference or improvement in
change-detection accuracy has yet been demonstrated by this addition.
