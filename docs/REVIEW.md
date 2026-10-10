# Implementation review — 2026-10-10

Reviewed the testing console, its API connections, upload and job recovery, input validation, ingestion, reconstruction/registration contracts, semantic evidence, exports, and the existing test coverage. This is a source and functional review, not proof that every possible input or robot deployment is correct.

## Verified issues corrected

- Browser storage access could throw during startup or run loading. Storage exceptions and invalid JSON shapes now have safe fallbacks. Session-only reviews remain exportable when durable browser storage is blocked, and the UI explains that limitation.
- Slow trial responses could replace a newer selection. A request generation gate prevents stale responses from committing UI state. Loading and processing lock trial inputs, and the form payload is captured before controls are disabled.
- File inputs could retain an earlier selection while the preview showed another completed trial. Reopened runs clear upload selections; empty file selections clear validity and previews. The viewed run is explicitly identified above the world viewer.
- Unsaved review notes disappeared when switching decisions. Drafts now survive selection and reload when browser storage permits it. Drafts are separate from saved assessments and are excluded from review counts and committed-review exports.
- Restoring a research run could select an unavailable profile on a core-only installation. Existing results retain their provenance while new uploads use available profiles.
- Missing evidence images and detector scores now produce explicit unavailable states rather than a broken inspector or an unqualified blank image.
- The processing profile overflowed at 320px. Controls now wrap; the small-screen and high-DPI browser checks pass. Pointer panning accounts for display pixel ratio.
- Upload cancellation, filesystem failure and worker submission failure could leave a queued job blocking subsequent submissions. Terminal failure handling now releases the queue, preserves failure status when storage permits, and closes both upload files.
- `changed_regions.ply` included `UNCHANGED` decisions. New exports include only ADDED, REMOVED and MOVED candidate locations, including both movement endpoints. UNKNOWN and UNCHANGED are excluded. Existing immutable run artifacts were not rewritten.
- PLY output could silently truncate rows when colors did not match points. Mismatched colors and nonfinite coordinates/colors are rejected.
- Infinite/NaN camera parameters and nonrigid depth transforms could pass input validation. Calibration values must be finite and depth extrinsics must contain a proper rigid rotation.
- A partially written completion report could be mistaken for a finished run after restart. Reports now become visible through an atomic file replacement. An interrupted publication regression verifies failure status and report-unavailable behavior.
- Recent-trial times now say when a report was opened in the browser. They are not presented as measured server completion times. Legacy browser timestamps remain readable.

## Verification

- **79 Python tests passed**, including synthetic end-to-end reconstruction and registration, original input/dataset checks, and new failure/export validation regressions.
- Ruff undefined-name/import checks passed across source, scripts and tests.
- Desktop/mobile browser replay passed using actual data from an existing completed synthetic run: reports, geometry, masks, reviews, JSON downloads, keyboard controls, trial restore and simulated server-busy recovery.
- Ten browser edge-case groups passed: invalid storage shape, draft persistence, stale requests, unavailable profiles, cleared uploads, busy controls, missing evidence, 320px/high-DPI layout, blocked storage and offline startup.
- Simulated failures and report replay do not establish new inference accuracy. Historical reports, datasets, checkpoint files and training strategy remain intact. No training or robot commands were initiated.

The idle local app was restarted to load the backend fixes. Health and all four historical completed reports were checked successfully after restart; no active jobs were interrupted.

See [browser replay](testing_console_smoke.json), [edge-case evidence](console_edge_cases.json), and [testing workflow](TESTING_UI.md).

## Remaining limits

The interface is suitable for local perception testing. Real smartphone before/after accuracy has not been measured against a verified change benchmark. Reviews remain browser-local rather than a shared company review service. The app has no robot control connection. Research model and dataset licensing must match the intended use. 3RScan access is still pending, Replica geometry remains incomplete, and a clean Docker build has not been verified in this environment. These are outstanding validation/integration limits, not completed features.
