"""Pinned local SAM 2 image masks and audited optical-flow mask propagation."""

from pathlib import Path
import os, sys, subprocess, hashlib
import cv2, numpy as np

SOURCE_COMMIT = "2b90b9f5ceec907a1c18123530e92e794ad901a4"


class Segmenter:
    def __init__(self, revision, device="cpu"):
        from huggingface_hub import HfApi, hf_hub_download
        import torch

        os.environ.setdefault("HF_HOME", "/workspace/spatialmemory-models/hf-home")
        os.environ.setdefault("HF_XET_CACHE", "/workspace/spatialmemory-models/xet")
        source = Path("/workspace/spatialmemory-thirdparty/sam2")
        actual = subprocess.check_output(
            ["git", "-C", str(source), "rev-parse", "HEAD"], text=True
        ).strip()
        if actual != SOURCE_COMMIT:
            raise ValueError("SAM 2 source revision mismatch")
        if subprocess.run(["git", "-C", str(source), "diff", "--quiet"]).returncode:
            raise ValueError("SAM 2 source has unreviewed tracked changes")
        sys.path.insert(0, str(source))
        from sam2.build_sam import build_sam2
        from sam2.sam2_image_predictor import SAM2ImagePredictor

        model = "facebook/sam2.1-hiera-tiny"
        filename = "sam2.1_hiera_tiny.pt"
        path = hf_hub_download(
            model,
            filename,
            revision=revision,
            cache_dir="/workspace/spatialmemory-models",
        )
        info = HfApi().model_info(model, revision=revision, files_metadata=True)
        expected = next(f.lfs.sha256 for f in info.siblings if f.rfilename == filename)
        digest = hashlib.sha256()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(1024 * 1024), b""):
                digest.update(chunk)
        if digest.hexdigest() != expected:
            raise ValueError("SAM 2 checkpoint integrity mismatch")
        torch.set_num_threads(4)
        self.predictor = SAM2ImagePredictor(
            build_sam2(
                "configs/sam2.1/sam2.1_hiera_t.yaml",
                path,
                device=device,
                apply_postprocessing=False,
            )
        )
        self.provenance = {
            "model": model,
            "revision": revision,
            "source_commit": SOURCE_COMMIT,
            "checkpoint_sha256": expected,
            "device": device,
            "calibrated": False,
        }

    def set_image(self, rgb):
        self.predictor.set_image(rgb)

    def mask(self, box):
        import torch

        with torch.inference_mode():
            masks, scores, _ = self.predictor.predict(
                box=np.asarray(box, float), multimask_output=True
            )
        best = int(np.argmax(scores))
        mask = masks[best].astype(bool)
        return mask, {
            "raw_predicted_iou": float(scores[best]),
            "area_pixels": int(mask.sum()),
            "provenance": self.provenance,
        }


def propagate_mask(previous, current, mask, tolerance=1.5):
    """Dense backward warp, forward/backward flow check; invalid regions are unknown."""
    if previous.shape != current.shape or mask.shape != previous.shape[:2]:
        raise ValueError("Flow/mask dimensions must agree")
    a = cv2.cvtColor(previous, cv2.COLOR_BGR2GRAY)
    b = cv2.cvtColor(current, cv2.COLOR_BGR2GRAY)
    params = (0.5, 3, 21, 4, 7, 1.5, 0)
    forward = cv2.calcOpticalFlowFarneback(a, b, None, *params)
    backward = cv2.calcOpticalFlowFarneback(b, a, None, *params)
    y, x = np.mgrid[: a.shape[0], : a.shape[1]].astype(np.float32)
    mx = x + backward[:, :, 0]
    my = y + backward[:, :, 1]
    warped = cv2.remap(mask.astype(np.uint8), mx, my, cv2.INTER_NEAREST).astype(bool)
    reverse_at_source = cv2.remap(forward, mx, my, cv2.INTER_LINEAR)
    error = np.linalg.norm(backward + reverse_at_source, axis=2)
    valid = (
        (error <= tolerance)
        & (mx >= 0)
        & (my >= 0)
        & (mx < a.shape[1] - 1)
        & (my < a.shape[0] - 1)
    )
    supported = warped & valid
    ratio = float(supported.sum() / max(1, warped.sum()))
    return supported, {
        "valid_fraction": ratio,
        "requires_reinitialization": ratio < 0.7,
        "camera_compensation": "not a physical motion measurement",
        "unknown_pixels": int((warped & ~valid).sum()),
    }
