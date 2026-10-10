# Comparison workspace

The existing local app now provides a responsive interface for engineers evaluating scene changes. Open the cloud environment's forwarded **port 8000**, or `http://127.0.0.1:8000/` when running locally. The current server reads the dashboard on each request, so refreshing is sufficient; no training or server restart was required.

## Comparison workflow

1. Name the comparison, enter a reviewer, and choose before/after videos of the same environment. Each upload is limited to 512 MiB. Slow camera passes and overlapping viewpoints help reconstruction.
2. Select Fast reconstruction or Detailed reconstruction · research, with optional object masks. The interface displays server-reported processing stages and errors, without inventing progress percentages. The VGGT profile displays its noncommercial checkpoint notice.
3. Inspect alignment, layer controls, the shared point cloud, and change candidates. Select a candidate to inspect its actual supporting images and decision record. Drag to orbit, Shift-drag to pan, or use arrow keys and +/- when the viewer is focused. Zero resets the view.
4. Save an assessment and notes: evidence supports the proposal, evidence does not support it, or further investigation is needed. These are human reviews, not ground-truth annotations. A saved review never edits the model output.
5. Download the original JSON report and a separate review JSON. Recent completed comparisons can be reopened from the same browser.

## Persistence and scope

Comparison names, reviewer information, recent comparison references and assessments are stored in **this browser's local storage**. They are not shared between reviewers or synchronized with a company database. Export reviews before clearing browser storage. Unsaved edits are retained as separate drafts and do not count as reviewed assessments. If persistent storage is blocked, reviews stay available for the current page session and can still be exported. Original run files and reports remain on the processing server. This is a perception evaluation interface with no robot control connection.

Metric distances are not claimed for unscaled monocular reconstruction. Alignment failure is visible. Empty results and unknown decisions are preserved without inventing objects, measurements or acceptance scores. Research profile selection does not establish commercial licensing eligibility.

## Executed verification

`docs/testing_console_smoke.json` records browser replay against an existing completed synthetic run, including real report/geometry/evidence loading, review persistence, JSON exports, keyboard controls, layer toggles, recent comparison restore, simulated server-busy recovery, mobile overflow and JavaScript errors. No new inference or training was launched by this replay. Current regression suite: 79 passed. No real-world accuracy or robot deployment validation is claimed.

```bash
SPATIALMEMORY_EXISTING_JOB=418b37750b0c404cb326c3a265191480 node scripts/check_testing_console.cjs
```

Screenshots: [desktop](screenshots/testing-console-desktop.png), [mobile](screenshots/testing-console-mobile.png).

The [implementation review](REVIEW.md) records verified defects, fixes, regression evidence and remaining limits. Recent-comparison dates denote the first browser opening, not a measured server completion time.

The workspace defaults to a dark theme. The header control switches to a light theme and remembers your choice when browser storage is available. User-facing summaries explain the result; the full result record and exported report retain the original model explanation and provenance.
