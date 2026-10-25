# Current status — SpatialMemory-X Ultra

**Canonical living status.** Prefer this file over chronological `PROGRESS.md` /
historical test-count claims in older docs. Historical `reports/*.json` are
dated evidence, not automatically “current truth.”

| Field | Value |
|---|---|
| Date (UTC) | 2026-10-25 |
| Branch | `cursor/change-detection-acceptance-893f` |
| Base commit | `3c62a0494f899387b8533cde0a9cf98215e2613c` (`main` / last known published) |
| Working tree at baseline | Clean match to `3c62a04` before this branch |
| Hardware this run | Linux x86_64, 4 CPUs, ~15 GiB RAM, **no NVIDIA GPU**, ~247 GiB free disk |
| Python | 3.12.3 + local `.venv` from `requirements/cpu.lock.txt` |

## Acceptance criteria (real phone pair)

| Requirement | Status |
|---|---|
| Process two phone recordings | **Blocked here** — private videos absent on this VM |
| Shared 3D representation | Documented historically; not re-run here |
| Automatic added/removed/moved detection | **Not achieved** (last measured: 53 UNKNOWN, 0 confirmed) |
| Supporting evidence + accurate descriptions | Incomplete; local VLM historically failed structured validation |
| Upload-to-results UI | Code present; not re-validated end-to-end on this VM |
| Reproducible demo MP4 | Not present / not verified here |

## Verified on this VM (2026-10-25)

- Repository checkout at `3c62a04`; feature branch created without committing on `main`.
- Clean CPU install from `requirements/cpu.lock.txt` + editable package.
- Baseline regression suite (pre-fix): **442 passed, 1 warning** (~17 s).
- After identity fail-closed fix + new tests: **443 passed, 1 warning** (~15 s).
- Code inspection: `ObjectState.identity_confirmed` previously defaulted to `True` and
  `match_core_objects` used `getattr(..., True)`. **Confirmed unsafe default.**
  Fixed on this branch to default / getattr `False`; regression tests added.
- Registration failure returns placeholder identity transform only with
  `valid=False`; `pipeline.py` refuses invalid registration before change claims.
- Sparse `change.compare` still requires literal boolean free-space; pipeline does
  not supply free-space callbacks → geometric regions stay UNKNOWN.
- `physical_visibility_verified` defaults `False` in object-view selection audits.

## Failed / unmet acceptance (from measured historical reports — not re-run here)

- Fresh phone neural run ~993 s CPU: 53 UNKNOWN, zero confirmed changes
  (`docs/SUBMISSION.md`, `reports/fresh_phone_validation_20261015.json`).
- Sparse registration could pass while whole-scene / dense overlap remained weak.
- AFTER point/camera self-reprojection inconsistency reported (~38 px median on a
  problematic view); three-view SIFT cycles on AFTER keyframes failed.
- LoFTR pair diagnostics (externally reported / partial JSON in
  `reports/learned_pair_validation_20261018.json`): 0 accepted matches at score ≥0.8
  for AFTER pairs 1020–2040, 1020–1440, 1440–2040, 2040–2640. **Not reproduced here**
  (frames + checkpoint missing).
- Classical model-free UI: visual candidates exist; they are not verified 3D removals.

## Phase 3 matching reproduction — BLOCKED

Cannot reproduce LoFTR AFTER pairs or head-consistency diagnostics without private
artifacts. Exact missing paths (from `docs/CLAUDE_HANDOFF.md` and install layout):

### Input videos (BEFORE / AFTER — do not reverse)

- `/workspace/attachments/a5ac41bf-942c-4fce-bf9d-f376005009cc/before room.mp4`
- `/workspace/attachments/17db9d20-db27-4ef4-b195-933f2a812511/after  room.mp4`

### Saved runs / diagnostics

- `/workspace/spatialmemory-fresh-validation-20261015-r2/`
- `/workspace/final-validation-20261015/` (incl. expected `config.json`)
- `/workspace/final-dense-scene-20261015/`
- `/workspace/spatialmemory-validation-20261015.zip`
- `/workspace/spatialmemory-localization-fix-20261014/`
- `/workspace/localization-audit-20261014/`
- `/workspace/removal-gate-audit-20261014/result.json`
- `/workspace/structured-visual-2b-mixed-20261014/visual_changes.json`

### Models / third-party / venv

- `/workspace/spatialmemory-models/` (or `SPATIALMEMORY_MODELS` / HF cache with VGGT, SAM2, DINO, Grounding DINO)
- `/workspace/spatialmemory-thirdparty/` (pinned VGGT + SAM2 sources)
- `/workspace/spatialmemory-venv/` (prior neural venv; optional if rebuilt)
- LoFTR checkpoint `loftr_indoor_ds_new.ckpt` (HF `kornia/loftr` revision
  `0fb3b456cf176746709c38fa9cf80695a1c82dac`, SHA-256
  `be9ff88b323ec27889114719f668ae41aff7034b56a4c4acbd46b8b180b87ed3`)

Also absent: `/workspace/models`, `/workspace/thirdparty`, `~/.spatialmemory`.

**Capability gaps:** no CUDA GPU; torch/neural extras not installed in the CPU `.venv`
(by design for Phase 1). Neural smoke and Phase 3+ need weights + preferably GPU.

## Changes on this branch (so far)

1. Fail-closed identity confirmation defaults (`semantics.py`, `identity.py`) + tests.
2. This `CURRENT_STATUS.md`.

Gates and thresholds were **not** relaxed. No accuracy claim follows from the
442 CPU regressions.

## Next actions (in brief order)

1. **Unblock:** transfer private BEFORE/AFTER videos, at least one saved neural run
   with AFTER keyframes, LoFTR checkpoint, and preferably VGGT/SAM stacks.
2. Phase 3: reproduce four AFTER LoFTR pairs with IMREAD_GRAYSCALE, full masks,
   score histograms, overlays — leave production 0.8 gate unchanged.
3. Phase 5–8: geometry → registration → identity → visibility on real evidence.
4. Keep classical candidates as a separate review mode (Phase 9).
5. UI/demo MP4 only after automatic evidence exists (Phase 10–12).
