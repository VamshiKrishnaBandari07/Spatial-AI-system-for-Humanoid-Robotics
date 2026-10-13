"""Download the pinned local VLM; verify the published checkpoint SHA256."""
import hashlib
import os
from pathlib import Path

os.environ.setdefault('HF_XET_CACHE', '/workspace/spatialmemory-models/xet')
from huggingface_hub import snapshot_download

folder = Path(snapshot_download(
    'HuggingFaceTB/SmolVLM-500M-Instruct',
    revision='a7da5b986cb59b408707209984f360a5f4ad7e47',
    cache_dir='/workspace/spatialmemory-models',
    allow_patterns=['*.json', '*.txt', '*.safetensors'], ignore_patterns=['onnx/*']))
with (folder / 'model.safetensors').open('rb') as stream:
    actual = hashlib.file_digest(stream, 'sha256').hexdigest()
expected = 'd05b567eeaf534e83d375551f068ed57b5f52d37c657197f644af5ef9db091a2'
if actual != expected:
    raise RuntimeError('Local VLM checkpoint checksum mismatch')
print(f'Verified local VLM checkpoint: {folder}')
