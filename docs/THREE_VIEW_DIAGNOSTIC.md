# Three-view geometry diagnostic

`spatialmemory.three_view_diagnostic.triangulate_and_validate_three_view`
triangulates supplied correspondences using views A and B and evaluates the
result against A, B, and withheld C. It is a standalone diagnostic; it does not
replace reconstruction, establish object identity, or classify removals.

Supply finite native-pixel observations `(N, 2)`, matching native-pixel pinhole
intrinsics `(3, 3)`, world-to-camera poses `(4, 4)`, and corresponding direct-head
world points `(N, 3)`. All poses and direct points must share one world frame and
scale. Undistort observations consistently before calling if needed.

`triangulation_valid_mask` reports geometric validity; `final_valid_mask` also
requires reprojection errors strictly below the configured limit in every view.
Reprojection rejection reasons refer to original input indices, including when
all points are rejected. Invalid geometric points retain NaN coordinates/errors.
Malformed inputs or zero baseline raise ValueError; rejected individual tracks
return masks rather than raising an all-points-failed exception.

Run synthetic regression tests:

```bash
python -m pytest -q tests/test_three_view_diagnostic.py
```

Limitations: feature extraction, correspondence verification, static-scene
verification, and native-grid conversion are caller responsibilities. Uncertainty
is uncalibrated. The implementation retains numerical baseline/homogeneous
cutoffs and is not invariant over unlimited scales. Returned NumPy arrays and
NaNs require explicit serialization handling. The mean 3D divergence uses the
geometric-valid subset, not exclusively tracks that pass held-out reprojection.
No real-video improvement or change-detection accuracy is established by these
synthetic tests. The production pipeline does not invoke this module yet.
