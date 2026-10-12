"""Map native masks into the exact upstream VGGT resize/crop/pad coordinates."""

import cv2
import numpy as np


def native_to_geometry_matrix(native_shape, geometry_shape, mode):
    """Pixel-center affine transform matching VGGT resize, crop and padding.

    Native intrinsics are A^-1 @ K_geometry. Both pixel coordinates use centers;
    transforms map from original image coordinates into the model input.
    """
    height, width = native_shape[:2]
    if mode == "pad" and height > width:
        nh, nw = 518, round(width * 518 / height / 14) * 14
    else:
        nw, nh = 518, round(height * 518 / width / 14) * 14
    if min(nh, nw) <= 0 or mode not in ("pad", "crop"):
        raise ValueError("Unsupported geometry preprocessing or aspect ratio")
    removed = (nh - 518) // 2 if mode == "crop" and nh > 518 else 0
    mapped_height = min(nh, 518) if removed else nh
    dh, dw = geometry_shape[0] - mapped_height, geometry_shape[1] - nw
    if dh < 0 or dw < 0:
        raise ValueError("Native mapping disagrees with geometry dimensions")
    sx, sy = nw / width, nh / height
    return np.array(
        [
            [sx, 0, dw // 2 + (sx - 1) / 2],
            [0, sy, dh // 2 - removed + (sy - 1) / 2],
            [0, 0, 1],
        ],
        dtype=float,
    )


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
