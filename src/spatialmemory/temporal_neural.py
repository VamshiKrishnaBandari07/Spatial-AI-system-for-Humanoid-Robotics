"""Real neural re-detection adapter for temporal tracking."""

import cv2
import numpy as np
from PIL import Image
from .appearance import masked_crop
from .detection_selection import select_detections


def neural_redetector(experts, config):
    def detect(image):
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        detections = experts.detector.infer(Image.fromarray(rgb), config.object_prompt)
        indices = select_detections(
            detections["boxes"],
            detections["raw_scores"],
            detections["labels"],
            config.detection_threshold,
            config.max_detections_per_frame,
        )
        experts.segmenter.set_image(rgb)
        results = []
        h, w = image.shape[:2]
        for index in indices:
            box = np.rint(
                np.clip(detections["boxes"][index], [0, 0, 0, 0], [w, h, w, h])
            ).astype(int)
            x0, y0, x1, y1 = box
            if x1 <= x0 or y1 <= y0:
                continue
            mask, audit = experts.segmenter.mask(box)
            inside = np.zeros((h, w), bool)
            inside[y0:y1, x0:x1] = True
            mask = mask & inside
            if mask.sum() < 16:
                continue
            appearance = experts.appearance.infer(
                Image.fromarray(masked_crop(rgb, mask, box))
            )
            results.append(
                {
                    "mask": mask,
                    "descriptor": appearance["descriptor"],
                    "label": detections["labels"][index],
                    "raw_detector_score": detections["raw_scores"][index],
                    "detector_provenance": detections["provenance"],
                    "appearance_provenance": appearance["provenance"],
                    "segmentation": audit,
                }
            )
        return results

    return detect
