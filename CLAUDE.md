# SpatialMemory-X Ultra — project instructions

Continue the existing implementation; inspect the working tree before editing. Do not replace the architecture, restart training, overwrite datasets/checkpoints, or stop unrelated processes. These instructions supply project context; they do not grant GitHub credentials or permission to publish changes.

## Read first

- `docs/COLAB.md` and `notebooks/SpatialMemory_Colab.ipynb` for the temporary upload workspace.
- `reports/APPLICATION_VERIFICATION.md` and `reports/colab_ui_verification.json` for execution evidence and limits.
- `docs/METHODOLOGY.md`, `docs/DATASET_ACCESS.md`, `MODEL_REGISTRY.md` and `EVALUATION.md` for architecture, approved data access, licences and evaluation.
- Check newer reports and commit history; historical test counts and status statements can be superseded. Run tests to determine current status.

## Commands

Use the existing virtual environment when present. On the original cloud workspace:

```bash
/workspace/spatialmemory-clean/bin/python -m pytest -q
/workspace/spatialmemory-venv/bin/python -m uvicorn spatialmemory.app:app --host 127.0.0.1 --port 8000
```

Check whether the application already occupies port 8000 before starting it. On a fresh local checkout with Python 3.11+ and FFmpeg/ffprobe available, use an isolated environment:

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -e '.[test]'
.venv/bin/python -m pytest -q
```

This prepares CPU regression testing, not neural checkpoints. Neural source/checkpoint setup retains pinned revisions and integrity verification; see `scripts/install_neural.sh` (original `/workspace` paths) or the Colab setup cell. Do not blindly execute cloud-specific paths on another machine.

## Integrity requirements

- The implemented neural profile uses VGGT, Grounding DINO, SAM 2 image masks and DINOv2. MASt3R/COLMAP integration, full video-memory tracking, deployment uncertainty calibration and validated navigation are not established.
- Default model loaders use CPU. GPU availability alone does not enable GPU inference.
- Phone profile: `configs/phone_challenge.json`. Keep multi-view identity gates, explicit unmatched assignments and bounded movement abstention. Do not lower thresholds simply to obtain positive detections.
- Reject unreliable registration; never silently use identity as a fallback. Monocular units are not metres without verified scale.
- UNKNOWN is insufficient evidence, not UNCHANGED. Detector/affinity/mask scores are not calibrated probabilities or measured accuracy.
- Use known changes only for independent evaluation, not as inference hints. Distinguish real, synthetic, partial-source and independently reviewed labels. Never invent annotations, accuracy, F1 or successful stages.
- Keep raw videos, restricted datasets, checkpoints and dataset-derived private imagery/clouds outside Git. Preserve original licence/access restrictions; existing approval does not authorize redistribution.
- Do not expose credentials, disable integrity checks, or bypass authentication. The Colab link is temporary and password protected; Colab execution requires the user's Google runtime.
- Checkpoint research licences do not establish commercial rights for humanoid company deployment.

## Current handoff

The Colab launch originally raised `NameError: ENV` when setup globals were absent. It now reconstructs its own configuration and checks setup before asking for a password. Fresh Google Colab installation/public relay execution remains user-side validation; local tests are not proof of that deployment.

The user-uploaded animated example completed fresh neural inference in approximately 308 seconds with 25 UNKNOWN decisions and no accepted cross-session object associations. This is execution evidence, not a real phone benchmark. Private files from the original cloud workspace are not included in a Git clone.

## Delivery

Run relevant tests, inspect actual reports/artifacts and report failures precisely. Preserve existing behavior unless a diagnosed issue justifies a change. Use a branch for new work. Before committing, inspect `git diff --check` and staged files for private data. Push or merge only within the user's explicit authorization; never force-push or disable permission prompts. Report exact files, checks, remaining blockers and commit hash.
