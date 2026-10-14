# Visual findings and 3D verification

The dashboard's **Visual change findings** section keeps two sources separate:

- Automated visual model output, labeled as model proposals.
- Manual visual review, labeled with its reviewer and explicitly not an automated
  prediction. It remains visible when the optional model is unavailable or fails.

Manual findings are stored per run in `reports/manual_visual_review.json` and
validated by `VisualReview` in `spatialmemory.visual_review`. They do not modify
`changes/changes.json`, spatial consensus, calibration, or benchmark predictions.
The schema requires before/after frame references with correct recording roles.
It does not independently verify the reviewer's visual interpretation.

A REMOVED visual finding is displayed as **Absent at previous location**: a
single local comparison cannot exclude relocation elsewhere. Empty added/moved
categories read **None verified in this review**, not “no changes occurred.”
3D location remains explicitly unverified. Manual findings are included in the
existing run artifact archive; no manual labels are injected into inference.

The current phone-pair review was saved only in its private run directory:
`/workspace/spatialmemory-localization-fix-20261014/reports/manual_visual_review.json`.
This code change does not deploy the UI or improve automated recognition.

Validation: seven visual API tests pass, including persistence of manual findings
when the model is unavailable and rejection of falsely marked automated reviews.
Dashboard JavaScript syntax was checked. No live browser deployment was tested.
