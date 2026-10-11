"""Select valid, nonduplicate proposals before applying the per-frame budget."""

import numpy as np
from .association import semantic_label


def select_detections(boxes, scores, labels, threshold, limit, iou_threshold=0.8):
    if (
        type(limit) is not int
        or limit < 1
        or not 0 <= threshold <= 1
        or not 0 < iou_threshold <= 1
    ):
        raise ValueError("Invalid detection selection thresholds or budget")
    if not len(boxes) == len(scores) == len(labels):
        raise ValueError("Detector output lengths disagree")
    selected = []
    valid = []
    for index, (box, score) in enumerate(zip(boxes, scores)):
        box = np.asarray(box, float)
        if (
            box.shape != (4,)
            or not np.isfinite(box).all()
            or not np.isfinite(score)
            or not 0 <= score <= 1
            or np.any(box[2:] <= box[:2])
        ):
            raise ValueError("Invalid detection box or score")
        if score >= threshold:
            valid.append(index)
    for index in sorted(valid, key=lambda i: scores[i], reverse=True):
        box = np.asarray(boxes[index], float)
        duplicate = False
        for previous in selected:
            if semantic_label(labels[index]) != semantic_label(labels[previous]):
                continue
            other = np.asarray(boxes[previous], float)
            intersection = np.prod(
                np.maximum(
                    0, np.minimum(box[2:], other[2:]) - np.maximum(box[:2], other[:2])
                )
            )
            union = (
                np.prod(box[2:] - box[:2])
                + np.prod(other[2:] - other[:2])
                - intersection
            )
            if intersection / union >= iou_threshold:
                duplicate = True
                break
        if not duplicate:
            selected.append(index)
            if len(selected) >= limit:
                break
    return selected
