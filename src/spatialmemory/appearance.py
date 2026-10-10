"""Pixel-coordinate crop contract shared by ingestion and appearance tests."""

import numpy as np


def masked_crop(rgb, mask, box):
    rgb = np.asarray(rgb)
    mask = np.asarray(mask)
    if rgb.ndim != 3 or rgb.shape[2] != 3 or mask.shape != rgb.shape[:2]:
        raise ValueError("RGB and mask pixel dimensions must agree")
    raw = np.asarray(box)
    if (
        raw.shape != (4,)
        or not np.isfinite(raw).all()
        or not np.equal(raw, np.rint(raw)).all()
    ):
        raise ValueError("Integer pixel box required")
    x0, y0, x1, y1 = raw.astype(int)
    h, w = rgb.shape[:2]
    if not (0 <= x0 < x1 <= w and 0 <= y0 < y1 <= h):
        raise ValueError("Box escapes image or is degenerate")
    crop = rgb[y0:y1, x0:x1].copy()
    crop[~mask[y0:y1, x0:x1].astype(bool)] = 0
    return crop
