# Dense shared-scene export

The main pipeline exports sparse feature clouds used in registration. For a
closer inspection of its computed VGGT surfaces, export a separate dense preview:

```bash
OMP_NUM_THREADS=4 python -m spatialmemory.dense_export \
  --run /path/to/completed/run --output /path/to/new/dense-output
```

This command supports completed quality-diverse VGGT runs with accepted positive
Sim(3) registration. It reads saved neural point maps, poses, intrinsics and RGB
frames. It does not rerun the neural models. It filters padding, applies the
existing point/depth confidence policy and strictly checks native-image
self-reprojection below the configured pixel limit. Default stride is two pixels.

After points are transformed into the before coordinate frame using the saved
AFTER-to-BEFORE registration. Outputs include color PLYs (`before.ply`,
`after.ply`, `aligned_scene.ply`), `diagnostics.json`, and a self-contained
`visualization/index.html` with before/after toggles. The viewer displays sampled
points for responsiveness; the PLY files contain the full exported cloud.

For browsers that restrict local files, serve the output directory:

```bash
python -m http.server 8080 --directory /path/to/dense-output
```

The preview is geometry-only and does not fabricate object-change markers.
Distances remain in unverified reconstruction units. Dense nearest-neighbor
coverage is diagnostic: differing coverage, predicted geometry errors and
registration errors can all reduce it. It is not change-detection accuracy,
metric calibration, or independent confirmation of camera geometry.

Keep generated scene data and the original recordings outside Git.
