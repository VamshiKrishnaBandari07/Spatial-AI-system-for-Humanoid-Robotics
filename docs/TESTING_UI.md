# Comparison workspace

The existing local app now provides a responsive interface for engineers evaluating scene changes. Run the app locally with `python -m uvicorn spatialmemory.app:app --host 127.0.0.1 --port 8000`. Dashboard content is read on each request; backend changes require a server restart. The cloud onboarding UI does not provide a public localhost preview.

## Comparison workflow

1. Name the comparison, enter a reviewer, and choose before/after videos of the same environment. Each upload is limited to 512 MiB. Slow camera passes and overlapping viewpoints help reconstruction.
2. Select Neural reconstruction or Geometry only, with optional object analysis. Neural reconstruction is selected initially when available. The interface displays server-reported processing stages and errors, without inventing progress percentages. The VGGT profile displays its noncommercial checkpoint notice.
3. Inspect alignment, layer controls, the shared point cloud, and change candidates. Select a candidate to inspect its actual supporting images and decision record. Drag to orbit, Shift-drag to pan, or use arrow keys and +/- when the viewer is focused. Zero resets the view.
4. Save an assessment and notes: evidence supports the proposal, evidence does not support it, or further investigation is needed. These are human reviews, not ground-truth annotations. A saved review never edits the model output.
5. Download the original JSON report and a separate review JSON. Recent completed comparisons can be reopened from the same browser.

## Persistence and scope

Comparison names, reviewer information, recent comparison references and assessments are stored in **this browser's local storage**. They are not shared between reviewers or synchronized with a company database. Export reviews before clearing browser storage. Unsaved edits are retained as separate drafts and do not count as reviewed assessments. If persistent storage is blocked, reviews stay available for the current page session and can still be exported. Original run files and reports remain on the processing server. This is a perception evaluation interface with no robot control connection.

Metric distances are not claimed for unscaled monocular reconstruction. Alignment failure is visible. Empty results and unknown decisions are preserved without inventing objects, measurements or acceptance scores. Research profile selection does not establish commercial licensing eligibility.

## Executed verification

`docs/testing_console_smoke.json` records browser replay against an existing completed synthetic run, including real report/geometry/evidence loading, review persistence, JSON exports, keyboard controls, layer toggles, recent comparison restore, simulated server-busy recovery, mobile overflow and JavaScript errors. No new inference or training was launched by this replay. Historical regression suite at that check: 79 passed. No real-world accuracy or robot deployment validation is claimed.

```bash
SPATIALMEMORY_EXISTING_JOB=418b37750b0c404cb326c3a265191480 node scripts/check_testing_console.cjs
```

Screenshots: [desktop](screenshots/testing-console-desktop.png), [mobile](screenshots/testing-console-mobile.png).

The [implementation review](REVIEW.md) records verified defects, fixes, regression evidence and remaining limits. Recent-comparison dates denote the first browser opening, not a measured server completion time.

The workspace defaults to a dark theme. The header control switches to a light theme and remembers your choice when browser storage is available. User-facing summaries explain the result; the full result record and exported report retain the original model explanation and provenance.

## Delivery checks — 11 October 2026

The dashboard upload route now uses quality/diversity frame selection, four neural frames per recording, multi-view object association and bounded movement decisions. No error budget is invented: movement remains Unknown without verified bounds. Geometry-only remains available if neural dependencies are missing. Ambiguous object labels and real-world association accuracy remain unresolved.

Completed comparisons provide job-scoped downloads for before/after clouds, the aligned scene, change decisions and alignment diagnostics. Only explicit artifact names are served; missing files and paths outside the run are rejected. Browser decoder failures display a preview message without claiming server decoding succeeded. MP4/MOV/MKV decoding is validated by ingestion during processing.

122 Python tests pass, including all previous 120. Browser edge-case checks pass on replay of the actual Chess identical-recording control: review persistence, load races, missing evidence, simulated busy/offline responses, blocked browser storage and 320px layout. The delivery browser check validates the neural default, actual PLY download, preview failure/recovery, Unknown wording and 375px layout with no JavaScript errors. Real dataset imagery and screenshots remain outside Git. This UI replay does not launch inference or establish change-detection accuracy.

Reproduce against a completed local job:

```bash
SPATIALMEMORY_URL=http://127.0.0.1:8000 \
SPATIALMEMORY_EXISTING_JOB=YOUR_COMPLETED_JOB_ID \
SPATIALMEMORY_UI_CHECK_OUTPUT=/workspace/ui-check \
node scripts/check_delivery_ui.cjs
```

This is a local research evaluation workspace. Commercial model licensing, shared reviewer storage, authenticated multi-user hosting and robot deployment are separate requirements.
