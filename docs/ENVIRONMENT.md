# Current cloud machine audit

Date: 2026-10-09. Checkout: `/workspace/Spatial-AI-system-for-Humanoid-Robotics`. Initial repository had no source files, tracked files or commits; Git remote access returned no refs.

- x86_64 Linux, five exposed Intel Xeon Platinum 8573C CPU cores.
- 33 GiB RAM, no swap; initially approximately 30 GiB free disk.
- Python 3.12.14, FFmpeg/ffprobe 7.1.5.
- No exposed nvidia-smi/GPU at audit. PyTorch 2.6.0 CPU installed in the optional-model development environment.
- Primary virtualenv: `/workspace/spatialmemory-venv` initially uses available system packages. Clean independent CPU virtualenv: `/workspace/spatialmemory-clean`, installed from pinned CPU requirements and verified.
- Public Git and package indexes work. Official GitHub README/license reads work. Hugging Face initially returned proxy HTTP 403; after the user updated settings, model cards/checkpoints downloaded successfully and five pretrained CPU models produced functional synthetic outputs, including SAM 2 masks.
- Model domains saved to draft: huggingface.co, *.huggingface.co, *.hf.co, dl.fbaipublicfiles.com. Existing package-manager presets retained.
- No credentials are required by the CPU workflow. No secret values were inspected, copied or saved. No private recordings/checkpoints exist in the project tree.
- Docker CLI exists, but container builds/runtime have not been validated.

Snapshots retain files/dependencies, not server processes. Use saved startup instructions for each new task. No new Git worktree is needed.

Current semantic profile: `configs/consensus_cpu.json`; official SAM 2 dependency checkout is `/workspace/spatialmemory-thirdparty/sam2` at commit `2b90b9f5ceec907a1c18123530e92e794ad901a4`. Neural installer repeated successfully after adding Hydra/iopath dependencies. Playwright and `/usr/bin/chromium` exercised the complete local upload/evidence/mobile workflow. Clean CPU suite: 53 passed. Neither a fresh published snapshot nor Docker runtime has been validated.
