"""Download pinned public research checkpoints via verified HTTPS; no remote code."""

import os

os.environ.setdefault("HF_HOME", "/workspace/spatialmemory-models/hf-home")
os.environ.setdefault("HF_XET_CACHE", "/workspace/spatialmemory-models/xet")
from huggingface_hub import snapshot_download, hf_hub_download
from neural_smoke import SPECS

for identifier, revision, purpose in SPECS:
    snapshot_download(
        identifier,
        revision=revision,
        cache_dir="/workspace/spatialmemory-models",
        allow_patterns=["*.json", "*.safetensors", "*.txt", "*.model"],
        ignore_patterns=["pytorch_model.bin"],
    )
hf_hub_download(
    "facebook/VGGT-1B",
    "model.pt",
    revision="860abec7937da0a4c03c41d3c269c366e82abdf9",
    cache_dir="/workspace/spatialmemory-models",
)
print(
    "Pinned model files available locally; VGGT checkpoint is research-only, not commercially licensed."
)

hf_hub_download(
    "facebook/sam2.1-hiera-tiny",
    "sam2.1_hiera_tiny.pt",
    revision="de431c4043854a71d8101e17995dfe596bf101a5",
    cache_dir="/workspace/spatialmemory-models",
)
