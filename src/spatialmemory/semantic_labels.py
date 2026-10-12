"""Pinned image/text neural label proposals, with explicit ambiguity margins."""

from typing import Any, Sequence
from PIL import Image
import numpy as np

MODEL_ID = "google/siglip-base-patch16-224"
MODEL_REVISION = "7fd15f0689c79d79e38b1c2e2e2370a7bf2761ed"
DEFAULT_LABELS = (
    "cup",
    "bottle",
    "perfume bottle",
    "spray can",
    "toilet paper roll",
    "tissue box",
    "bowl",
    "plate",
    "pan",
    "bag",
    "backpack",
    "chair",
    "table",
    "cabinet",
    "book",
    "plant",
    "monitor",
    "phone",
    "laptop",
    "box",
    "kettle",
    "jar",
    "food container",
)


class SemanticLabeler:
    """Contrastive rankings are NOT calibrated class probabilities or ground truth."""

    def __init__(
        self,
        cache_dir: str,
        labels: Sequence[str] = DEFAULT_LABELS,
        device: str = "cpu",
    ) -> None:
        import torch
        from transformers import AutoProcessor, SiglipModel

        if len(labels) < 2 or len(set(labels)) != len(labels):
            raise ValueError("At least two distinct semantic labels required")
        self.labels = tuple(labels)
        self.device = device
        self.processor = AutoProcessor.from_pretrained(
            MODEL_ID,
            revision=MODEL_REVISION,
            cache_dir=cache_dir,
            trust_remote_code=False,
        )
        self.model = (
            SiglipModel.from_pretrained(
                MODEL_ID,
                revision=MODEL_REVISION,
                cache_dir=cache_dir,
                trust_remote_code=False,
            )
            .to(device)
            .eval()
        )
        inputs = self.processor(
            text=[f"a photo of a {label}." for label in self.labels],
            padding="max_length",
            max_length=64,
            return_tensors="pt",
        ).to(device)
        with torch.inference_mode():
            text = self.model.get_text_features(**inputs)
            self.text_features = text / text.norm(dim=-1, keepdim=True)

    def infer(
        self, image: Image.Image, minimum_margin: float = 0.015
    ) -> dict[str, Any]:
        import torch

        if not 0 <= minimum_margin <= 1:
            raise ValueError("Invalid semantic ranking margin")
        inputs = self.processor(images=image, return_tensors="pt").to(self.device)
        with torch.inference_mode():
            features = self.model.get_image_features(**inputs)
            features = features / features.norm(dim=-1, keepdim=True)
            similarities = (features @ self.text_features.T).cpu().numpy()[0]
        if not np.isfinite(similarities).all():
            raise ValueError("Nonfinite semantic model output")
        order = np.argsort(-similarities)
        margin = float(similarities[order[0]] - similarities[order[1]])
        return {
            "label": self.labels[order[0]]
            if margin >= minimum_margin
            else "unidentified object",
            "top_candidates": [
                {"label": self.labels[i], "cosine": float(similarities[i])}
                for i in order[:3]
            ],
            "ranking_margin": margin,
            "minimum_margin": minimum_margin,
            "calibrated": False,
            "provenance": {
                "model": MODEL_ID,
                "revision": MODEL_REVISION,
                "device": self.device,
            },
        }
