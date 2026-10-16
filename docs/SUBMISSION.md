# Submission: SpatialMemory-X Ultra

A research prototype for comparing before/after indoor videos using neural 3D
reconstruction, semantic perception and visibility-aware evidence aggregation.

## Demonstrated result

The latest submitted phone pair completed a fresh, uncached inference in
992.90 seconds on CPU. VGGT reconstruction and neural object perception ran.
The exported shared-frame geometry contains 37,297 before points and 22,435 after
points after filtering. Its scale is unverified; coordinates are not metres.

The interactive geometry viewer was tested in Chromium, including recording
layer toggles, with no JavaScript errors. The regression suite passed 333 tests.

## Acceptance status

| Requested outcome | Verified status |
|---|---|
| Process two phone recordings | Completed on the submitted pair |
| Generate shared 3D representation | Exported; whole-scene alignment and metric scale remain unverified |
| Automatically identify added, removed and moved objects | **Not achieved on this pair: 53 inconclusive decisions, zero confirmed changes** |
| Accurate automatic change descriptions | Not demonstrated; tested local VLM outputs failed validation |
| Public source repository and reproducible workflow | Code, documentation and aggregate reports provided |

Sparse registration passed its existing checks: 20/24 correspondence inliers,
3/4 held-out inliers, held-out RMSE 0.01255 reconstruction units. However, only
17.50% of the dense after points were within the configured registration threshold
of the before cloud. Partial coverage and reconstruction errors affect this
measurement; it is not detection accuracy or independent alignment ground truth.

The pink container is absent from its earlier location in manually inspected
frames. That observation is not an automated model prediction and is not counted
as a successful detection benchmark. No precision, recall or F1 claim is made.

## Review and reproduce

1. Follow the root README for installation or the Colab workflow. Hosted UI
   availability has not been verified; the repository itself is not an inference
   server.
2. Inspect `reports/fresh_phone_validation_20261015.json` for the measured run.
3. Run the CLI with explicit before/after paths and a saved configuration:

   ```bash
   spatialmemory before.mp4 after.mp4 --config config.json --output results/new_run
   ```

4. Follow [dense export instructions](DENSE_SCENE_EXPORT.md) to inspect computed
   geometry. The dense exporter reuses a completed run's point maps; it does not
   claim a second neural inference.
5. Read [failure audit](PIPELINE_FAILURE_AUDIT.md) and
   [local visual backend results](LOCAL_VISUAL.md) before interpreting predictions.

The private validation ZIP contains the interactive dense viewer, colored PLYs,
configuration, actual reports, logs and a preview. Raw videos and checkpoints are
excluded. Private scene artifacts are not committed to the public repository.

## Submission description

> SpatialMemory-X Ultra implements a reproducible pipeline for two-video indoor
> reconstruction, object perception, association and visibility-aware change
> reasoning. It includes a shared-scene viewer and diagnostic reports. The current
> real-video validation demonstrates pipeline execution and geometry export, but
> does not yet demonstrate reliable automatic added/removed/moved detection.

This is suitable for submission as a documented research prototype, not as a
completed or accuracy-validated change-detection solution.
