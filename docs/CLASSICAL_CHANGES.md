# Model-free visual change candidates

This optional command uses OpenCV only: SIFT mutual ratio matching, RANSAC
homography alignment with held-out checks, robust Lab brightness normalization,
morphological segmentation, and connected components. It does not load neural
models, modify production classifications, or assign semantic names.

```bash
PYTHONPATH=src python -m spatialmemory.classical_changes \
  --run /workspace/spatialmemory-fresh-validation-20261015-r2 \
  --output /workspace/new-classical-review --max-frames 24
```

Input is an existing ingestion run with before/after timestamped manifests and
native frames. Output must be a new directory. No new reconstruction is performed.
Up to three after frames join each before-anchor viewpoint group only if alignment
passes checks: 15 fitting inliers, at least 60% held-out residuals under 2 native
pixels, median held-out error below 2 pixels, and an inlier hull covering at least
8% of the before image. This is a planar-image check, not proof of whole-scene 3D
alignment. Time sampling does not guarantee independent viewpoints.

Open index.html to inspect comparisons. Each pair folder includes before.jpg,
after_native.jpg, after_aligned.jpg, mask.png, comparison.jpg and candidate crops.
report.json records timestamps, transforms, alignment diagnostics, region bounds,
and counts of overlapping candidate boxes in other accepted pairs of that group.
These support counts are not independent confirmation or tracking identities.
Groups overlap; region totals are not counts of distinct changed objects.

## Actual saved-phone-frame test (2026-10-24)

24 frames per recording produced six viewpoint groups, seven accepted pairs, and
42 candidate regions. The automatically extracted region at before:241 bbox
[243,260,370,398] covers the pink handled container's old location, compared with
after:2640. Human inspection of that exported crop sees the container before and
the background bottle afterward. The algorithm outputs only an unlabeled visual
change candidate; no physical removal or 3D localization is established.

Shadows, mirrors, parallax, and alignment differences also generate candidates.
There is no measured precision/recall or guarantee that every change is captured.
The current HTML is a standalone evidence gallery, not integration into the main
upload UI. Private images are kept outside Git.
