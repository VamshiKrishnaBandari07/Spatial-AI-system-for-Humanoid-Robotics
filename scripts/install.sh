#!/usr/bin/env bash
set -euo pipefail
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
if [ -z "${SPATIALMEMORY_WORKSPACE:-}" ]; then
  if [ -d /workspace ] && [ -w /workspace ]; then SPATIALMEMORY_WORKSPACE=/workspace; else SPATIALMEMORY_WORKSPACE="$HOME/.spatialmemory"; fi
fi
mkdir -p "$SPATIALMEMORY_WORKSPACE"
command -v ffmpeg >/dev/null
command -v ffprobe >/dev/null
SPATIALMEMORY_VENV="${SPATIALMEMORY_VENV:-$SPATIALMEMORY_WORKSPACE/spatialmemory-venv}"
python3 -m venv "$SPATIALMEMORY_VENV"
"$SPATIALMEMORY_VENV/bin/python" -m pip install -r requirements/cpu.lock.txt
"$SPATIALMEMORY_VENV/bin/python" -m pip install --no-build-isolation --no-deps -e .
"$SPATIALMEMORY_VENV/bin/python" -m pip check
"$SPATIALMEMORY_VENV/bin/python" -m pytest -q
