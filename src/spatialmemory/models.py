"""Local neural experts. No remote video APIs, no remote Python model code."""

from dataclasses import dataclass
import re


@dataclass
class ModelSpec:
    identifier: str
    revision: str
    purpose: str


class Expert:
    def __init__(self, spec: ModelSpec, cache_dir, device="cpu"):
        if not re.fullmatch(r"[0-9a-fA-F]{40}", spec.revision):
            raise ValueError("Use an immutable model revision for reproducibility")
        self.spec = spec
        self.cache_dir = str(cache_dir)
        self.device = device
        self.processor = None
        self.model = None
        self.temperature = None

    def load(self):
        import os
        from pathlib import Path

        os.environ.setdefault("HF_HOME", str(Path(self.cache_dir) / "hf-home"))
        os.environ.setdefault("HF_XET_CACHE", str(Path(self.cache_dir) / "xet"))
        from transformers import (
            AutoImageProcessor,
            AutoModel,
            AutoModelForDepthEstimation,
            AutoProcessor,
            AutoModelForZeroShotObjectDetection,
        )

        kwargs = dict(
            revision=self.spec.revision,
            cache_dir=self.cache_dir,
            trust_remote_code=False,
        )
        if self.spec.purpose == "appearance":
            self.processor = AutoImageProcessor.from_pretrained(
                self.spec.identifier, **kwargs
            )
            self.model = AutoModel.from_pretrained(self.spec.identifier, **kwargs)
        elif self.spec.purpose == "relative_depth":
            self.processor = AutoImageProcessor.from_pretrained(
                self.spec.identifier, **kwargs
            )
            self.model = AutoModelForDepthEstimation.from_pretrained(
                self.spec.identifier, **kwargs
            )
        elif self.spec.purpose == "detection":
            self.processor = AutoProcessor.from_pretrained(
                self.spec.identifier, **kwargs
            )
            self.model = AutoModelForZeroShotObjectDetection.from_pretrained(
                self.spec.identifier, **kwargs
            )
        else:
            raise ValueError("Unsupported expert purpose")
        self.model.to(self.device).eval()
        return self

    def infer(self, image, prompt=None):
        if self.model is None:
            raise RuntimeError("Load the expert first")
        import torch

        if self.spec.purpose == "detection" and not prompt:
            raise ValueError("Open-vocabulary detection needs a prompt")
        kwargs = {"images": image, "return_tensors": "pt"}
        if prompt:
            kwargs["text"] = prompt
        inputs = self.processor(**kwargs).to(self.device)
        with torch.inference_mode():
            result = self.model(**inputs)
        provenance = {
            "model": self.spec.identifier,
            "revision": self.spec.revision,
            "device": self.device,
        }
        if self.spec.purpose == "appearance":
            vector = result.last_hidden_state[:, 0].cpu().numpy()[0]
            from .association import unit_descriptor

            vector = unit_descriptor(vector)
            return {"descriptor": vector.tolist(), "provenance": provenance}
        if self.spec.purpose == "relative_depth":
            depth = result.predicted_depth.cpu().numpy()[0]
            return {
                "relative_depth": depth,
                "metric": False,
                "visibility_usable": False,
                "provenance": provenance,
            }
        detections = self.processor.post_process_grounded_object_detection(
            result,
            inputs.input_ids,
            threshold=0.3,
            text_threshold=0.25,
            target_sizes=[(image.height, image.width)],
        )[0]
        # Grounding DINO's postprocessor retains queries whose maximum token
        # sigmoid exceeds 0.3. Preserve those exact pre-sigmoid logits; never
        # reconstruct logits from rounded scores or label similarity.
        query_logits = result.logits[0].max(dim=-1).values
        retained_logits = query_logits[query_logits.sigmoid() > 0.3].cpu().tolist()
        if len(retained_logits) != len(detections["scores"]):
            raise ValueError("Detector logits/postprocessing correspondence mismatch")
        return {
            "boxes": detections["boxes"].cpu().tolist(),
            "raw_scores": detections["scores"].cpu().tolist(),
            "raw_logits": retained_logits,
            "calibrated_scores": self.temperature.probabilities(
                retained_logits, self.spec.revision
            ).tolist() if self.temperature is not None else None,
            "calibration_source": self.temperature.source if self.temperature is not None else None,
            "labels": detections.get("text_labels", [])
            if "text_labels" in detections
            else detections.get("labels", []),
            "calibrated": self.temperature is not None,
            "provenance": provenance,
        }


class VGGTExpert:
    """Requires reviewed official VGGT source installed separately and a local checkpoint."""

    def __init__(self, checkpoint, device="cpu"):
        import torch
        from vggt.models.vggt import VGGT

        self.torch = torch
        self.device = device
        self.checkpoint = str(checkpoint)
        self.model = VGGT().to(device).eval()
        self.model.load_state_dict(
            torch.load(checkpoint, map_location="cpu", weights_only=True)
        )

    def infer(self, paths, preprocessing="pad"):
        from vggt.utils.load_fn import load_and_preprocess_images
        from vggt.utils.pose_enc import pose_encoding_to_extri_intri

        images = load_and_preprocess_images(
            [str(p) for p in paths], mode=preprocessing
        ).to(self.device)
        with self.torch.inference_mode():
            predictions = self.model(images)
        extrinsics, intrinsics = pose_encoding_to_extri_intri(
            predictions["pose_enc"], images.shape[-2:]
        )
        return {
            "extrinsics": extrinsics.cpu().numpy(),
            "intrinsics": intrinsics.cpu().numpy(),
            "depth": predictions["depth"].cpu().numpy(),
            "world_points": predictions["world_points"].cpu().numpy(),
            "world_points_conf": predictions["world_points_conf"].cpu().numpy(),
            "depth_conf": predictions["depth_conf"].cpu().numpy(),
            "scale_status": "unverified",
            "checkpoint": self.checkpoint,
        }
