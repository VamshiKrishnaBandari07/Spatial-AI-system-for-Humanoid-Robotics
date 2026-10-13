# Rotation-aware crop recovery

`oriented_detection.detect_orientations` invokes the existing pinned Grounding
DINO model at four quarter-turns. Boxes are mapped to original image coordinates
before IoU suppression. Raw predictions are retained. Rotations count as one
view, and scores are uncalibrated. Competing labels must not become verified
semantic identities merely because one has the highest raw score.

Run on an existing completed inference (output must be a new directory):

```bash
OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 python -m spatialmemory.crop_recheck \
  --run /path/to/completed/run --output /path/to/new/recovery \
  --rotation-aware --max-views 2
```

Run detection on a single image:

```bash
python -m spatialmemory.oriented_detection \
  --image /path/to/crop.jpg --output /path/to/new/output \
  --cache /path/to/model-cache --prompt 'bottle. cup. cabinet.'
```

The real experiment reused the existing prompt unchanged and evaluated two saved
crops, not complete videos. Eight neural calls took 49.34 seconds. The original
after-crop pass had zero accepted detections. The rotation ensemble recovered a
white bottle proposal (raw score 0.597), along with cabinet and false book
proposals. It also incorrectly labeled the source pink container as a table.
This is evidence of a crop/orientation recognition weakness, not successful
change detection. See `reports/rotation_crop_recovery.json`.

`context_alignment.fit_context` is a separate experimental diagnostic. It masks
out the source object and fits a local background homography, rejecting
extrapolation, poor held-out reprojection, and inadequate image coverage. All 45
after views were rejected for the tested source observation. Therefore it is
not used to replace the 3D projection or infer empty space. A homography cannot
generally model parallax between a protruding object and its background.

These checks retain the 2-pixel limit. The next unresolved task is validating
object geometry and camera correspondence before interpreting missing detections
as empty space. This change provides no removal, movement, or addition accuracy.

Validation: `python -m pytest -q` — 325 passed, 1 dependency deprecation warning.
Private crops and model weights remain outside Git.
