# Ray–box visibility integration

`semantics.visibility_evidence` now attaches `raybox_diagnostics` to each evaluated
view, using the actual object extent, target reconstruction depth grid, intrinsics,
world-to-camera pose and existing depth-support mask. The repository registration
maps AFTER to BEFORE; the adapter explicitly inverts it for AFTER target views.
No RGB-to-depth resize is applied: K and depth are from the same reconstruction grid.
Every depth pixel is evaluated to avoid subsampling away small projected objects.

The new evidence is diagnostic only. It does not replace the existing consensus
or turn an empty former location into a removal. Extents estimated from partial
observations are not guaranteed complete object volumes.

## Supplying uncertainty

Optional per-surface `raybox_uncertainty` metadata contains:

- `depth_error_bound`: scalar or H×W nonnegative absolute camera-Z error.
- `geometry_error_bound`: conservative absolute camera-Z geometry error.
- `scale_id`: the shared target reconstruction/depth scale identifier.
- `source`: provenance establishing the bounds and their applicability.

These fields are caller supplied; a source string alone does not validate them.
Without them, the adapter reports `missing_validated_uncertainty_bounds`. Neural
confidence scores, head agreement and registration RMSE are not error bounds.
Unverified monocular distances remain reconstruction units, not metres.

`conformal_depth_bound` computes the finite-sample order statistic
ceil((n+1)(1-alpha)) from held-out frame-level maximum absolute depth errors.
It requires independently calibrated reference depth and exchangeability with
the test frame. It raises when there are too few calibration frames. Its coverage
is marginal under those assumptions, not a universal or deterministic guarantee.
Correlated pixels must not be counted as independent calibration examples.

`propagated_geometry_bound` propagates supplied deterministic localization,
translation, rotation and scale bounds via triangle inequalities and
||R-Rhat|| <= 2 sin(theta/2). The resulting Euclidean bound also bounds camera-Z
error. It does not establish lateral ray support under pose uncertainty; that
requires spatial uncertainty handling before physical change decisions can use it.
Neither utility estimates unknown bounds from unlabeled monocular predictions.

## Diagnostics and validation

Each view logs transform direction, depth convention and grid dimensions. With
bounds present it includes sampled ray pixel coordinates, near/far depths, state
codes, counts and projected-region bounds. Counts use all intersecting pixels;
only the JSON ray list is capped at 64. Boundary contact is a warning, not proof of
complete object coverage. No-intersection rays do not prove physical absence.

A check on saved real phone-video VGGT depth/K/poses exercised the transform
adapter and reported missing bounds. It did not generate free-space evidence or
run fresh neural inference. Synthetic tests cover transforms, Sim(3), clipping,
parallel rays, scale metadata and uncertainty intervals. They are not accuracy
benchmarks. The first adapter test exposed stride-4 sampling missing a small box;
the adapter now uses stride 1.
