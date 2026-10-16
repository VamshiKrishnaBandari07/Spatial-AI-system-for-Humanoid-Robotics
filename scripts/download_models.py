"""Download pinned public research checkpoints via verified HTTPS; no remote code."""

import argparse
import hashlib
from pathlib import Path

from spatialmemory.paths import configure_hf_cache, models_dir
configure_hf_cache()
from huggingface_hub import snapshot_download, hf_hub_download
from neural_smoke import SPECS

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument(
    "--multiview",
    action="store_true",
    help="Include the independent semantic label model",
)
args = parser.parse_args()

for identifier, revision, purpose in SPECS:
    snapshot_download(
        identifier,
        revision=revision,
        cache_dir=str(models_dir()),
        allow_patterns=["*.json", "*.safetensors", "*.txt", "*.model"],
        ignore_patterns=["pytorch_model.bin"],
    )
hf_hub_download(
    "facebook/VGGT-1B",
    "model.pt",
    revision="860abec7937da0a4c03c41d3c269c366e82abdf9",
    cache_dir=str(models_dir()),
)

if args.multiview:
    from huggingface_hub import HfApi
    from spatialmemory.semantic_labels import MODEL_ID, MODEL_REVISION

    info = HfApi().model_info(MODEL_ID, revision=MODEL_REVISION, files_metadata=True)
    if info.card_data.license != "apache-2.0":
        raise RuntimeError("Semantic model license differs from reviewed metadata")
    snapshot_download(
        MODEL_ID,
        revision=MODEL_REVISION,
        cache_dir=str(models_dir()),
        allow_patterns=["*.json", "*.safetensors", "*.model", "*.txt"],
    )
    checkpoint = Path(
        hf_hub_download(
            MODEL_ID,
            "model.safetensors",
            revision=MODEL_REVISION,
            cache_dir=str(models_dir()),
        )
    )
    expected = next(
        file.lfs.sha256
        for file in info.siblings
        if file.rfilename == "model.safetensors"
    )
    digest = hashlib.sha256()
    with checkpoint.open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)
    if digest.hexdigest() != expected:
        raise RuntimeError("Semantic label checkpoint SHA-256 mismatch")
    print("Verified pinned multi-view semantic label checkpoint", expected)
print(
    "Pinned model files available locally; VGGT checkpoint is research-only, not commercially licensed."
)

hf_hub_download(
    "facebook/sam2.1-hiera-tiny",
    "sam2.1_hiera_tiny.pt",
    revision="de431c4043854a71d8101e17995dfe596bf101a5",
    cache_dir=str(models_dir()),
)
