#!/usr/bin/env bash
set -euo pipefail
cd /workspace/Spatial-AI-system-for-Humanoid-Robotics
SPATIALMEMORY_VENV="${SPATIALMEMORY_VENV:-/workspace/spatialmemory-venv}"
export HF_HOME=/workspace/spatialmemory-models/hf-home
export HF_XET_CACHE=/workspace/spatialmemory-models/xet
mkdir -p "$HF_HOME" "$HF_XET_CACHE"
"$SPATIALMEMORY_VENV/bin/python" -m pip install torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cpu
"$SPATIALMEMORY_VENV/bin/python" -m pip install -r requirements/neural.lock.txt
SPATIALMEMORY_VGGT_SOURCE=/workspace/spatialmemory-thirdparty/vggt
SPATIALMEMORY_VGGT_COMMIT=a288dd0f14786c93483e45524328726ab7b1b4ce
if [ ! -e "$SPATIALMEMORY_VGGT_SOURCE" ]; then
  mkdir -p "$SPATIALMEMORY_VGGT_SOURCE"
  git -C "$SPATIALMEMORY_VGGT_SOURCE" init
  git -C "$SPATIALMEMORY_VGGT_SOURCE" remote add origin https://github.com/facebookresearch/vggt.git
  git -C "$SPATIALMEMORY_VGGT_SOURCE" fetch --depth 1 origin "$SPATIALMEMORY_VGGT_COMMIT"
  git -C "$SPATIALMEMORY_VGGT_SOURCE" checkout --detach FETCH_HEAD
fi
[ "$(git -C "$SPATIALMEMORY_VGGT_SOURCE" rev-parse HEAD)" = "$SPATIALMEMORY_VGGT_COMMIT" ] || { echo 'Existing VGGT source differs; preserve it and inspect before updating.' >&2; exit 1; }
SPATIALMEMORY_SAM_SOURCE=/workspace/spatialmemory-thirdparty/sam2
SPATIALMEMORY_SAM_COMMIT=2b90b9f5ceec907a1c18123530e92e794ad901a4
if [ ! -e "$SPATIALMEMORY_SAM_SOURCE" ]; then
  mkdir -p "$SPATIALMEMORY_SAM_SOURCE"
  git -C "$SPATIALMEMORY_SAM_SOURCE" init
  git -C "$SPATIALMEMORY_SAM_SOURCE" remote add origin https://github.com/facebookresearch/sam2.git
  git -C "$SPATIALMEMORY_SAM_SOURCE" fetch --depth 1 origin "$SPATIALMEMORY_SAM_COMMIT"
  git -C "$SPATIALMEMORY_SAM_SOURCE" checkout --detach FETCH_HEAD
fi
[ "$(git -C "$SPATIALMEMORY_SAM_SOURCE" rev-parse HEAD)" = "$SPATIALMEMORY_SAM_COMMIT" ] || { echo 'Existing SAM 2 source differs; preserve it and inspect before updating.' >&2; exit 1; }
git -C "$SPATIALMEMORY_SAM_SOURCE" diff --quiet || { echo 'SAM 2 source has unreviewed changes.' >&2; exit 1; }
"$SPATIALMEMORY_VENV/bin/python" scripts/download_models.py
"$SPATIALMEMORY_VENV/bin/python" -m pip check
