"""Actual pinned DINOv2 crop replay, not an identity accuracy benchmark."""

import argparse
import json
from pathlib import Path
import cv2
import numpy as np
import torch
from PIL import Image
from spatialmemory.benchmarking import sha256
from spatialmemory.memory import Memory
from spatialmemory.models import Expert, ModelSpec


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    torch.set_num_threads(4)
    expert = Expert(
        ModelSpec(
            "facebook/dinov2-small",
            "ed25f3a31f01632728cabb09d1542f84ab7b0056",
            "appearance",
        ),
        "/workspace/spatialmemory-models",
    ).load()
    frames = {
        f["id"]: f["path"]
        for state in ["before", "after"]
        for f in json.loads((args.run / state / "manifest.json").read_text())["frames"]
    }
    memory = Memory(args.run / "spatial_memory.sqlite")
    try:
        obs = memory.observations()
    finally:
        memory.close()
    chosen = [
        o
        for state in ["before", "after"]
        for o in [x for x in obs if x.provenance["state"] == state][:4]
    ]
    rows = []
    cosine = lambda a, b: float(
        np.asarray(a) @ np.asarray(b) / (np.linalg.norm(a) * np.linalg.norm(b))
    )
    for o in chosen:
        rgb = cv2.cvtColor(cv2.imread(frames[o.frame_id]), cv2.COLOR_BGR2RGB)
        mask = (
            cv2.imread(
                str(
                    args.run / "evidence/supporting_frames" / o.provenance["mask_file"]
                ),
                cv2.IMREAD_GRAYSCALE,
            )
            > 0
        )
        x0, y0, x1, y1 = o.provenance["box"]
        crop = rgb[y0:y1, x0:x1].copy()
        local = mask[y0:y1, x0:x1]
        masked = crop.copy()
        masked[~local] = 0
        repeated = expert.infer(Image.fromarray(masked))["descriptor"]
        context = expert.infer(Image.fromarray(crop))["descriptor"]
        rows.append(
            {
                "observation_id": o.id,
                "source_frame_sha256": sha256(frames[o.frame_id]),
                "mask_sha256": sha256(
                    args.run / "evidence/supporting_frames" / o.provenance["mask_file"]
                ),
                "crop_width": x1 - x0,
                "crop_height": y1 - y0,
                "black_background_fraction": float(1 - local.mean()),
                "stored_vs_repeated_masked_cosine": cosine(o.descriptor, repeated),
                "masked_vs_unmasked_cosine": cosine(repeated, context),
                "repeated_norm": float(np.linalg.norm(repeated)),
            }
        )
    snapshot = (
        Path("/workspace/spatialmemory-models/models--facebook--dinov2-small/snapshots")
        / expert.spec.revision
    )
    weights = list(snapshot.glob("*.safetensors"))
    result = {
        "mode": "Actual DINOv2 model invocation on saved source crops; no new detector or segmentation invocation",
        "source_report_sha256": sha256(args.run / "report.json"),
        "revision": expert.spec.revision,
        "checkpoint_hashes": {p.name: sha256(p) for p in weights},
        "processor": expert.processor.to_dict(),
        "features": "Final-layer CLS token, L2 normalized; arithmetic mean of normalized descriptors and re-normalization across accepted views",
        "rows": rows,
        "independent_identity_labels": False,
        "threshold_recommendation": "No change supported: reviewed identity distributions are unavailable",
        "limitations": [
            "Central crop can omit parts of elongated objects",
            "Black masked background changes embedding context",
            "Sensitivity is measured; neither crop is assumed correct without independent identities",
        ],
    }
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(
        json.dumps(
            {
                "replayed": len(rows),
                "minimum_reproduction_cosine": min(
                    r["stored_vs_repeated_masked_cosine"] for r in rows
                ),
                "masked_context_cosines": [
                    r["masked_vs_unmasked_cosine"] for r in rows
                ],
            }
        )
    )


if __name__ == "__main__":
    main()
