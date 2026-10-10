#!/usr/bin/env bash
set -euo pipefail
cd /workspace/Spatial-AI-system-for-Humanoid-Robotics
command -v ffmpeg >/dev/null
command -v ffprobe >/dev/null
SPATIALMEMORY_VENV="${SPATIALMEMORY_VENV:-/workspace/spatialmemory-venv}"
python3 -m venv "$SPATIALMEMORY_VENV"
"$SPATIALMEMORY_VENV/bin/python" -m pip install -r requirements/cpu.lock.txt
"$SPATIALMEMORY_VENV/bin/python" -m pip install --no-build-isolation --no-deps -e .
"$SPATIALMEMORY_VENV/bin/python" -m pip check
"$SPATIALMEMORY_VENV/bin/python" -m pytest -q
