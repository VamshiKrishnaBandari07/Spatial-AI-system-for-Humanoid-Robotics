# Arbitrary before/after video workflow

Use any two overlapping indoor RGB videos. Private phone paths from prior
validation are optional; the UI and CLI accept ordinary local files.

## Upload UI (anytime testing)

```bash
# After scripts/install.sh (CPU) and optionally scripts/install_neural.sh
source /workspace/spatialmemory-venv/bin/activate   # or your .venv
python -m uvicorn spatialmemory.app:app --host 127.0.0.1 --port 8000
```

Open `http://127.0.0.1:8000/`:

1. Upload **before** and **after** videos (≤512 MiB each).
2. Choose **sparse** (no neural weights) or **Neural reconstruction** if
   `GET /health` reports `vggt` / `semantics` capabilities.
3. Optional: object analysis, multi-view object analysis, or classical
   model-free visual candidates (`/classical`).
4. Wait for stage progress; download report, evidence, ZIP, and 3D world.

The service is a **local single-worker** tool: no public auth. Bind to
localhost only. Concurrent jobs return HTTP 429.

## CLI

```bash
python -m spatialmemory_x run \
  --before /path/to/before.mp4 \
  --after /path/to/after.mp4 \
  --scene-id room_demo \
  --config configs/phone_challenge.json \
  --output results/room_demo_$(date +%Y%m%d_%H%M%S)
```

- Output directories are immutable; choose a new `--output` each run.
- Without neural install, use `configs/cpu.json` (sparse) or omit config.
- Manifests: `--manifest examples/scene_manifest.json --project-root .`
  (template paths must be replaced with real files).

## Classical-only review (no weights)

```bash
# Via UI: /classical
# Or after ingesting frames into a run directory:
PYTHONPATH=src python -m spatialmemory.classical_changes \
  --run /path/to/ingested_run \
  --output /path/to/new_classical_review --max-frames 24
```

Classical outputs are `VISUAL_CHANGE_CANDIDATE` only — not ADDED/REMOVED/MOVED.

## LoFTR pair diagnostics (when a completed neural run exists)

```bash
python -m spatialmemory.matching_diagnostics \
  --run /path/to/completed_run \
  --checkpoint /path/to/loftr_indoor_ds_new.ckpt \
  --output reports/loftr_after_pairs.json
```

Default AFTER index pairs: 1020–2040, 1020–1440, 1440–2040, 2040–2640.
Production score gate stays **0.8**; this tool only measures.

## What “working” means for arbitrary videos

- Pipeline completes or fails with an explicit reason (no silent identity registration).
- UNKNOWN means insufficient evidence, not “unchanged.”
- Confirmed ADDED/REMOVED/MOVED on a new pair requires measured evidence and
  independent review — do not treat a green UI run as acceptance.
