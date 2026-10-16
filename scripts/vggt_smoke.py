"""Bounded two-frame official VGGT CPU inference. Not a scene benchmark."""

import sys, json, time, resource, argparse
from pathlib import Path

from spatialmemory.paths import configure_hf_cache, models_dir, thirdparty_dir
configure_hf_cache()
SOURCE = str(thirdparty_dir() / "vggt")
sys.path.insert(0, SOURCE)
import torch, numpy as np
from huggingface_hub import hf_hub_download, HfApi
from spatialmemory.models import VGGTExpert


def main():
    p = argparse.ArgumentParser()
    p.add_argument("frames", nargs=2, type=Path)
    p.add_argument("--output", type=Path, default=Path("results/vggt_smoke"))
    a = p.parse_args()
    a.output.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(4)
    revision = "860abec7937da0a4c03c41d3c269c366e82abdf9"
    checkpoint = hf_hub_download(
        "facebook/VGGT-1B",
        "model.pt",
        revision=revision,
        cache_dir=str(models_dir()),
    )
    import hashlib

    info = HfApi().model_info(
        "facebook/VGGT-1B", revision=revision, files_metadata=True
    )
    expected = next(f.lfs.sha256 for f in info.siblings if f.rfilename == "model.pt")
    digest = hashlib.sha256()
    with open(checkpoint, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    if digest.hexdigest() != expected:
        raise ValueError("Checkpoint integrity mismatch")
    started = time.perf_counter()
    print("Loading verified official VGGT checkpoint", flush=True)
    expert = VGGTExpert(checkpoint)
    print("Running two-frame CPU inference", flush=True)
    result = expert.infer(a.frames)
    arrays = {k: v for k, v in result.items() if isinstance(v, np.ndarray)}
    for k, v in arrays.items():
        if not np.isfinite(v).all():
            raise ValueError(f"Nonfinite VGGT output: {k}")
    np.savez_compressed(a.output / "predictions.npz", **arrays)
    summary = {
        "model": "facebook/VGGT-1B",
        "checkpoint_revision": revision,
        "checkpoint_sha256": expected,
        "source_commit": "a288dd0f14786c93483e45524328726ab7b1b4ce",
        "device": "cpu",
        "frames": [str(f) for f in a.frames],
        "input_type": "synthetic textured panels",
        "status": "passed",
        "shapes": {k: list(v.shape) for k, v in arrays.items()},
        "seconds": time.perf_counter() - started,
        "peak_process_rss_mib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        / 1024,
        "scale_status": "unverified",
        "evaluation": "inference smoke only, no reconstruction accuracy or calibrated confidence claim",
        "checkpoint_license": "CC-BY-NC-4.0; research profile, no commercial deployment",
    }
    (a.output / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
