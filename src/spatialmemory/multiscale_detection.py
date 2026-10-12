"""Bounded native-resolution neural crop inference; no heuristic class labels."""

from typing import Any, Protocol
from PIL import Image
import numpy as np


class Detector(Protocol):
    def infer(self, image: Image.Image, prompt: str) -> dict[str, Any]: ...


def multiscale_detect(
    detector: Detector,
    image: Image.Image,
    prompt: str,
    tile_size: int = 512,
    max_tiles: int = 4,
    max_refinements: int = 0,
) -> dict[str, Any]:
    """Run full-frame and bounded overlapping tiles, mapping boxes to native pixels.

    Scores remain raw model scores. Near-identical boxes across passes are
    collapsed by score; alternate labels and crop provenance are preserved.
    No scene-specific coordinates or user-reported changes are used.
    """
    if tile_size < 128 or max_tiles < 1:
        raise ValueError("Invalid multiscale detection bounds")
    width, height = image.size
    if max_refinements < 0:
        raise ValueError("Negative refinement budget")
    if max(width, height) <= tile_size and not max_refinements:
        return detector.infer(image, prompt)
    xs = np.unique(
        np.linspace(
            0,
            max(0, width - tile_size),
            max(1, int(np.ceil(width / (tile_size * 0.75)))),
        )
        .round()
        .astype(int)
    )
    ys = np.unique(
        np.linspace(
            0,
            max(0, height - tile_size),
            max(1, int(np.ceil(height / (tile_size * 0.75)))),
        )
        .round()
        .astype(int)
    )
    tiles = [
        (
            int(x),
            int(y),
            min(width, int(x) + tile_size),
            min(height, int(y) + tile_size),
        )
        for y in ys
        for x in xs
    ]
    if len(tiles) > max_tiles:
        tiles = [
            tiles[i]
            for i in np.unique(
                np.linspace(0, len(tiles) - 1, max_tiles).round().astype(int)
            )
        ]
    calls = [(0, 0, width, height), *[rect for rect in tiles if rect != (0, 0, width, height)]]
    candidates: list[dict[str, Any]] = []
    base: dict[str, Any] | None = None
    for pass_id, rectangle in enumerate(calls):
        x, y, x1, y1 = rectangle
        result = detector.infer(
            image if pass_id == 0 else image.crop(rectangle), prompt
        )
        if base is None:
            base = result
        boxes, scores, labels = result["boxes"], result["raw_scores"], result["labels"]
        if not len(boxes) == len(scores) == len(labels):
            raise ValueError("Detector output lengths disagree")
        for detection_index, (box, score, label) in enumerate(zip(boxes, scores, labels)):
            box = np.asarray(box, float)
            if (
                box.shape != (4,)
                or not np.isfinite(box).all()
                or not np.isfinite(score)
            ):
                raise ValueError("Invalid neural detection")
            box = np.clip(box, [0, 0, 0, 0], [x1 - x, y1 - y, x1 - x, y1 - y]) + [
                x,
                y,
                x,
                y,
            ]
            if np.any(box[2:] <= box[:2]):
                continue
            candidates.append(
                {
                    "box": box,
                    "score": float(score),
                    "label": str(label),
                    "pass_id": pass_id,
                    "logit": result.get("raw_logits", [None] * len(boxes))[detection_index],
                    "probability": (result.get("calibrated_scores") or [None] * len(boxes))[detection_index],
                }
            )
    # Secondary candidate-centred native crop pass. The detector resizes each
    # crop using its own trained preprocessing; no pixels or labels are invented.
    if max_refinements < 0:
        raise ValueError("Negative refinement budget")
    small = [row for row in candidates if
             np.prod(row["box"][2:] - row["box"][:2]) / (width * height) < 0.02
             or row["score"] < 0.5]
    refined_rectangles: list[tuple[int, int, int, int]] = []
    for row in sorted(small, key=lambda r: r["score"], reverse=True):
        if len(refined_rectangles) >= max_refinements:
            break
        box = row["box"]
        center = (box[:2] + box[2:]) / 2
        half = np.maximum((box[2:] - box[:2]), 48)
        rect = tuple(map(int, [max(0, np.floor(center[0] - half[0])),
                              max(0, np.floor(center[1] - half[1])),
                              min(width, np.ceil(center[0] + half[0])),
                              min(height, np.ceil(center[1] + half[1]))]))
        if rect in refined_rectangles:
            continue
        refined_rectangles.append(rect)
        x, y, x1, y1 = rect
        result = detector.infer(image.crop(rect), prompt)
        if not len(result["boxes"]) == len(result["raw_scores"]) == len(result["labels"]):
            raise ValueError("Refinement detector output lengths disagree")
        for detection_index, (local, score, label) in enumerate(zip(result["boxes"], result["raw_scores"], result["labels"])):
            local = np.asarray(local, float)
            if local.shape != (4,) or not np.isfinite(local).all() or not np.isfinite(score):
                raise ValueError("Invalid refinement detection")
            native = np.clip(local, [0, 0, 0, 0], [x1-x, y1-y, x1-x, y1-y]) + [x, y, x, y]
            if np.all(native[2:] > native[:2]):
                candidates.append({"box": native, "score": float(score),
                                   "label": str(label), "pass_id": len(calls) + len(refined_rectangles)-1,
                                   "logit": result.get("raw_logits", [None] * len(result["boxes"]))[detection_index],
                                   "probability": (result.get("calibrated_scores") or [None] * len(result["boxes"]))[detection_index]})
    candidates.sort(key=lambda item: item["score"], reverse=True)
    selected: list[dict[str, Any]] = []
    for candidate in candidates:
        duplicate = None
        for previous in selected:
            a, b = candidate["box"], previous["box"]
            inter = float(
                np.prod(
                    np.maximum(0, np.minimum(a[2:], b[2:]) - np.maximum(a[:2], b[:2]))
                )
            )
            union = float(np.prod(a[2:] - a[:2]) + np.prod(b[2:] - b[:2]) - inter)
            if union > 0 and inter / union >= 0.9:
                duplicate = previous
                break
        if duplicate is not None:
            duplicate["alternate_labels"].append(candidate["label"])
        else:
            selected.append({**candidate, "alternate_labels": []})
    assert base is not None
    return {
        "boxes": [row["box"].tolist() for row in selected],
        "raw_scores": [row["score"] for row in selected],
        "labels": [row["label"] for row in selected],
        "raw_logits": [row["logit"] for row in selected],
        "calibrated_scores": [row["probability"] for row in selected] if base.get("calibrated") else None,
        "calibration_source": base.get("calibration_source"),
        "calibrated": bool(base.get("calibrated")),
        "provenance": {
            **base["provenance"],
            "multiscale": True,
            "crop_rectangles": [list(rect) for rect in calls],
            "refinement_rectangles": [list(rect) for rect in refined_rectangles],
        },
        "alternate_labels": [row["alternate_labels"] for row in selected],
        "detection_passes": [row["pass_id"] for row in selected],
    }
