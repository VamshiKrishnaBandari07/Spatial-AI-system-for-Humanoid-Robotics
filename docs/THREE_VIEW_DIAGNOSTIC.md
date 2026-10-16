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

## Saved real-video diagnostic, 2026-10-16

Reproduce without modifying predictions or fitting cameras:

```bash
PYTHONPATH=src python scripts/diagnose_three_view_run.py \
  --run /workspace/spatialmemory-fresh-validation-20261015-r2 \
  --output /workspace/three-view-new-diagnostic
```

The output directory must not exist. This runner uses SIFT (4,000-feature cap),
mutual Lowe-ratio matching at 0.7 and exact three-view descriptor-index cycles.
These filters reject ambiguity but do not establish physical identity or static
landmarks. No withheld reprojection errors are used to select tracks.

The AFTER saved keyframes yielded zero three-view candidate cycles. The strongest
BEFORE triplet had 69 candidates, 35 passing A/B reprojection, and zero passing C;
median withheld error was 3.564 native pixels. Another single-track triplet passed,
which does not validate scene geometry. Direct C-head errors are self-reprojection
checks and must not be mistaken for independent multi-view accuracy.

Searching all 45 saved AFTER frames for a bridge between after:1020 and after:2040
found at most two shared intermediate feature indices (after:1440). That is not
sufficient for a robust geometry-recovery claim. Dense temporal tracking or a
learned correspondence backend should be tested next; replacing keyframes requires
fresh inference and new cache provenance. Do not reuse these camera arrays with
new frame indices.

Aggregate results: reports/real_three_view_validation.json and
reports/real_three_view_bridge_search.json. Private match overlays and per-track
arrays: /workspace/three-view-real-20261016-r2. No change classifications were
produced or altered by this diagnostic.
