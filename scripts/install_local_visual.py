"""Download the pinned local VLM; verify the published checkpoint SHA256."""
import hashlib
from pathlib import Path

from spatialmemory.paths import configure_hf_cache, models_dir
configure_hf_cache()
from huggingface_hub import snapshot_download

from argparse import ArgumentParser
from spatialmemory.local_visual import MODEL_PROFILES
parser = ArgumentParser(description=__doc__)
parser.add_argument('--model-profile', choices=MODEL_PROFILES, default='500m')
args = parser.parse_args()
identifier, revision, expected = MODEL_PROFILES[args.model_profile]
folder = Path(snapshot_download(identifier, revision=revision,
    cache_dir=str(models_dir()),
    allow_patterns=['*.json', '*.txt', '*.jinja', 'model.safetensors']))
with (folder / 'model.safetensors').open('rb') as stream:
    actual = hashlib.file_digest(stream, 'sha256').hexdigest()
if actual != expected:
    raise RuntimeError('Local VLM checkpoint checksum mismatch')
print(f'Verified local VLM checkpoint: {folder}')
