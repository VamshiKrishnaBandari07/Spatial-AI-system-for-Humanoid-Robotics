"""Held-out sensor-depth agreement of actual saved VGGT arrays; not probability calibration."""

import argparse
import json
from pathlib import Path
import cv2
import numpy as np
from spatialmemory.contracts import Config, Frame
from spatialmemory.inputs import load_manifest
from spatialmemory.ingestion import neural_keyframes
from spatialmemory.sensor_geometry import reconstruct_sensor
from spatialmemory.benchmarking import sha256


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run", type=Path, required=True)
    p.add_argument("--manifest", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    report = json.loads((args.run / "report.json").read_text())
    cfg = Config.model_validate(report["config"])
    _, files, _ = load_manifest(args.manifest, Path.cwd())
    rows = []
    for state in ["before", "after"]:
        frames = [
            Frame.model_validate(f)
            for f in json.loads((args.run / state / "manifest.json").read_text())[
                "frames"
            ]
        ]
        native = reconstruct_sensor(frames, files[state])
        selected = neural_keyframes(frames, cfg.neural_frames_per_video)
        with np.load(
            args.run / "neural" / state / "predictions.npz", allow_pickle=False
        ) as d:
            for i, f in enumerate(selected):
                measured = native.surfaces[f.id]["depth"]
                H, W = measured.shape
                nh = round(H * (518 / W) / 14) * 14
                measured = cv2.resize(
                    measured, (518, nh), interpolation=cv2.INTER_NEAREST
                )
                if nh > 518:
                    measured = measured[(nh - 518) // 2 : (nh - 518) // 2 + 518]
                predicted = d["depth"][0, i, :, :, 0]
                conf = d["world_points_conf"][0, i]
                if measured.shape != predicted.shape:
                    raise ValueError("Model/sensor preprocessing mismatch")
                valid = (
                    np.isfinite(measured)
                    & (measured > 0)
                    & np.isfinite(predicted)
                    & (predicted > 0)
                    & (conf >= np.quantile(conf, 0.5))
                )
                x = predicted[valid]
                y = measured[valid]
                if len(x) < 100:
                    raise ValueError(
                        "Insufficient measured/neural depth comparison support"
                    )
                scale = float(np.median(y[::2] / x[::2]))
                residual = np.abs(scale * x[1::2] - y[1::2])
                rows.append(
                    {
                        "state": state,
                        "frame_id": f.id,
                        "depth_pairs": len(x),
                        "scale_fit_pixels": len(x[::2]),
                        "heldout_pixels": len(residual),
                        "fitted_monocular_scale": scale,
                        "heldout_median_absolute_depth_error_metres": float(
                            np.median(residual)
                        ),
                        "heldout_p90_absolute_depth_error_metres": float(
                            np.quantile(residual, 0.9)
                        ),
                        "heldout_median_relative_depth_error": float(
                            np.median(residual / y[1::2])
                        ),
                    }
                )
    result = {
        "inference_report_sha256": sha256(args.run / "report.json"),
        "rows": rows,
        "interpretation": "Native measured sensor depth versus actual saved neural depth after robust per-frame scale fitting; held-out alternating pixels are spatially correlated. Nearest-pixel depth resampling; no room-level accuracy or calibrated uncertainty claim.",
    }
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
