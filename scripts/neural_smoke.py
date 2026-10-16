"""Real pretrained CPU inference on an explicitly synthetic image; no accuracy claims."""


from spatialmemory.paths import configure_hf_cache, models_dir
configure_hf_cache()
from pathlib import Path
import argparse, json, time
import numpy as np
from PIL import Image
from spatialmemory.models import Expert, ModelSpec

SPECS = [
    ("facebook/dinov2-small", "ed25f3a31f01632728cabb09d1542f84ab7b0056", "appearance"),
    (
        "depth-anything/Depth-Anything-V2-Small-hf",
        "5426e4f0f36572d16453bbda7a8389317b1bef99",
        "relative_depth",
    ),
    (
        "IDEA-Research/grounding-dino-tiny",
        "a2bb814dd30d776dcf7e30523b00659f4f141c71",
        "detection",
    ),
]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("image", type=Path)
    p.add_argument("--output", type=Path, default=Path("results/neural_smoke.json"))
    a = p.parse_args()
    image = Image.open(a.image).convert("RGB")
    records = []
    a.output.parent.mkdir(parents=True, exist_ok=True)
    for identifier, revision, purpose in SPECS:
        t = time.perf_counter()
        print("Loading", identifier, flush=True)
        try:
            expert = Expert(
                ModelSpec(identifier, revision, purpose),
                str(models_dir()),
            ).load()
            result = expert.infer(
                image,
                prompt="chair. table. box. bottle." if purpose == "detection" else None,
            )
            summary = {
                "model": identifier,
                "revision": revision,
                "purpose": purpose,
                "status": "passed",
                "seconds": time.perf_counter() - t,
                "input": str(a.image),
                "evaluation": "inference smoke only; no real-world accuracy claim",
            }
            if purpose == "appearance":
                vector = np.asarray(result["descriptor"])
                assert np.isfinite(vector).all() and len(vector) == 384
                summary.update(
                    descriptor_dimensions=len(vector),
                    norm=float(np.linalg.norm(vector)),
                )
            elif purpose == "relative_depth":
                depth = result.pop("relative_depth")
                assert np.isfinite(depth).all() and depth.ndim == 2
                summary.update(depth_shape=list(depth.shape), finite=True, metric=False)
                np.save(a.output.with_suffix(".depth.npy"), depth)
            else:
                summary["detections"] = result
            records.append(summary)
            del expert
        except Exception as e:
            records.append(
                {
                    "model": identifier,
                    "revision": revision,
                    "status": "failed",
                    "error": str(e),
                    "type": type(e).__name__,
                }
            )
        print(json.dumps(records[-1]), flush=True)
        a.output.write_text(json.dumps(records, indent=2))
    if any(r["status"] != "passed" for r in records):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
