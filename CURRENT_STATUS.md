# Current status — SpatialMemory-X Ultra

**Canonical living status.** Prefer this file over chronological `PROGRESS.md` /
historical test-count claims in older docs. Historical `reports/*.json` are
dated evidence, not automatically “current truth.”

| Field | Value |
|---|---|
| Date (UTC) | 2026-10-25 |
| Acceptance PR branch | `cursor/change-detection-acceptance-893f` ([PR #1](https://github.com/VamshiKrishnaBandari07/Spatial-AI-system-for-Humanoid-Robotics/pull/1) — identity + status) |
| Engineering follow-up branch | `cursor/change-detection-eng-893f` |
| Base commit | `3c62a0494f899387b8533cde0a9cf98215e2613c` |
| Hardware | Linux x86_64, 4 CPUs, ~15 GiB RAM, **no NVIDIA GPU**, ~246 GiB free disk |
| Python | 3.12.3; CPU `.venv` + neural `/workspace/spatialmemory-venv` |

## Acceptance criteria (real phone pair)

| Requirement | Status |
|---|---|
| Process two phone recordings | **Blocked** — private MP4s not on this VM (not in Git) |
| Shared 3D representation | Code + neural stack ready; not re-measured on real pair here |
| Automatic added/removed/moved | **Not achieved** historically (53 UNKNOWN, 0 confirmed); not reclaimed |
| Supporting evidence + descriptions | Partial; VLM historically failed structured validation |
| Upload / UI for arbitrary videos | Documented in `docs/ARBITRARY_VIDEO.md`; app hardened this branch |
| Playable demo MP4 | Not recorded — needs real upload session after videos exist |

**Do not treat CI/regressions or a green health check as change-detection acceptance.**

## Verified on this VM (2026-10-25)

### Models / neural stack (fetchable — installed)

- `scripts/install.sh` + `scripts/install_neural.sh` completed.
- `GET`-equivalent `neural_capabilities()` → `{sparse: True, vggt: True, semantics: True}`.
- Third-party pins: VGGT `a288dd0…`, SAM2 `2b90b9f…` under `/workspace/spatialmemory-thirdparty/`.
- Checkpoints under `/workspace/spatialmemory-models/` including VGGT-1B, SAM2.1 tiny,
  DINOv2, Grounding DINO tiny, Depth Anything V2 small, SigLIP (multiview), LoFTR.
- LoFTR `loftr_indoor_ds_new.ckpt` SHA-256
  `be9ff88b323ec27889114719f668ae41aff7034b56a4c4acbd46b8b180b87ed3` **verified**.
- `pip check` clean on neural venv. Torch `2.6.0+cpu`.

### Code / tests

- Fail-closed identity confirmation (`ObjectState` / joint `getattr` → `False`) — PR #1.
- Classical honesty: HTML escape, border-region reject, `region_count_is_not_object_count`.
- LoFTR diagnostic module `spatialmemory.matching_diagnostics` (gates unchanged at 0.8).
- App: client error redaction + bounded in-memory job map prune.
- Docs: `docs/ARBITRARY_VIDEO.md` for anytime upload/CLI testing.
- Regression: **454 passed**, 1 Starlette/httpx warning (CPU `.venv`) after Bugbot fixes.
- Bugbot on head `160b22a`: **pass** (ingest lookup + match-cap agreement fixed).
- Smokes (synthetic, not acceptance): sparse pipeline OK; VGGT 2-frame OK (~52 s);
  classical OK; live LoFTR diagnose OK (`accepted_returned` agrees with production).

## Exact missing private videos (blocker for Phase 3 / 11)

These are **not** in GitHub. Place them at the known paths (or update this file with
new paths) then re-run Phase 3 immediately:

- `/workspace/attachments/a5ac41bf-942c-4fce-bf9d-f376005009cc/before room.mp4`
- `/workspace/attachments/17db9d20-db27-4ef4-b195-933f2a812511/after  room.mp4`

Also useful (not required to start LoFTR if fresh frames are extracted):

- `/workspace/spatialmemory-fresh-validation-20261015-r2/`
- `/workspace/final-validation-20261015/`
- `/workspace/spatialmemory-localization-fix-20261014/`
- `/workspace/final-dense-scene-20261015/`
- `/workspace/spatialmemory-validation-20261015.zip`
- `/workspace/removal-gate-audit-20261014/result.json`
- `/workspace/structured-visual-2b-mixed-20261014/visual_changes.json`
- `/workspace/localization-audit-20261014/`

## Phase 3 command (ready when AFTER frames exist)

```bash
source /workspace/spatialmemory-venv/bin/activate
python -m spatialmemory.matching_diagnostics \
  --run /path/to/completed_run_or_ingested_frames \
  --checkpoint /workspace/spatialmemory-models/models--kornia--loftr/snapshots/0fb3b456cf176746709c38fa9cf80695a1c82dac/loftr_indoor_ds_new.ckpt \
  --output reports/loftr_after_pairs_measured.json
```

Default pairs: 1020–2040, 1020–1440, 1440–2040, 2040–2640. Production threshold stays 0.8.

## Arbitrary-video testing (no private pair required)

See `docs/ARBITRARY_VIDEO.md`. Local UI:

```bash
source /workspace/spatialmemory-venv/bin/activate
python -m uvicorn spatialmemory.app:app --host 127.0.0.1 --port 8000
```

## Historical unmet results (not re-run)

- Phone neural ~993 s: 53 UNKNOWN, 0 confirmed (`docs/SUBMISSION.md`).
- Dense overlap weak vs sparse Sim(3) pass; AFTER head inconsistency reported.
- Classical candidates ≠ verified removals.

## Next

1. Obtain private BEFORE/AFTER MP4s at the paths above (or provide replacements).
2. Reproduce Phase 3 LoFTR diagnostics with measured score histograms.
3. Drive Phase 5–11 on real evidence without relaxing gates; record
   `/cursor/stores/self/media/upload-to-results-demo.mp4` after a real UI session.
