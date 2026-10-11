"""Map native masks into the exact upstream VGGT resize/crop/pad coordinates."""

import cv2
import numpy as np


def mask_to_geometry(mask, shape, mode):
    height, width = mask.shape
    if mode == "pad" and height > width:
        nh, nw = 518, round(width * 518 / height / 14) * 14
    else:
        nw = 518
        nh = round(height * 518 / width / 14) * 14
    if min(nh, nw) <= 0:
        raise ValueError("Image aspect ratio cannot be represented by VGGT")
    mapped = cv2.resize(
        mask.astype(np.uint8), (nw, nh), interpolation=cv2.INTER_NEAREST
    )
    if mode == "crop" and nh > 518:
        top = (nh - 518) // 2
        mapped = mapped[top : top + 518]
    elif mode != "pad" and mode != "crop":
        raise ValueError("Unsupported geometry preprocessing")
    dh, dw = shape[0] - mapped.shape[0], shape[1] - mapped.shape[1]
    if dh < 0 or dw < 0:
        raise ValueError("Mask mapping disagrees with processed geometry dimensions")
    return np.pad(mapped, ((dh // 2, dh - dh // 2), (dw // 2, dw - dw // 2))).astype(
        bool
    )
