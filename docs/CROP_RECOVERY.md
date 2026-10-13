# Object-directed crop recovery

`python -m spatialmemory.crop_recheck --run RESULTS --output NEW_DIRECTORY`
consumes a completed run's object-specific view rankings, converts projected
regions from reconstruction coordinates to native RGB using the exact preprocessing
affine transform, expands context, and runs fresh Grounding DINO detection.
Fresh DINOv2 crop embeddings compare the source observation with detected target
crops. Both sides use unmasked crops in this recovery pass, so comparisons do not
mix stored masked features with unmasked target features.

The default covers all eligible BEFORE tracks with two target views each.
`--max-objects` optionally limits the track budget; `--max-views` limits views.
Coverage metadata lists eligible, selected, deferred, and missing-view tracks.
The largest stored source observation is used instead of a tiny first fragment.
Repeated event records are deduplicated by track ID; physical tracks are not merged
without identity evidence. Selection uses no scene-specific labels or ground truth. Saved native patches and recheck.json contain
the actual detections, raw scores, similarities, and inherited occupancy diagnostics.
A proposal requires cosine >= .85 and a .05 advantage over the next candidate;
these thresholds are matching gates, not calibrated probabilities.

This implements active re-detection on ranked views, but does not yet establish
cross-object one-to-one identity or search the whole room for moved objects. It
produces recovery proposals for inspection, not confirmed changes. Missing
matches never imply removal. Recovered observations are not silently inserted
into the persistent gallery; they need joint assignment and geometric validation.
Existing 3D/visibility uncertainty blockers remain explicit. Reused geometry and
fresh perception are distinguished in the result.
