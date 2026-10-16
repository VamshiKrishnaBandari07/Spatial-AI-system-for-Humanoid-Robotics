"""Check native->model pixel mapping against VGGT's own preprocessing arithmetic.

``upstream_preprocess`` re-implements the resize/crop/pad logic of
``vggt.utils.load_fn.load_and_preprocess_images`` at the pinned commit
a288dd0f14786c93483e45524328726ab7b1b4ce (without torch; padding value is
irrelevant to positions). A bright marker is pushed through it and located by
centroid; ``native_to_geometry_matrix`` must predict that location. This is a
synthetic geometric consistency test, not a measure of model accuracy.
"""

import numpy as np
import pytest
from PIL import Image

from spatialmemory.pixel_mapping import native_to_geometry_matrix

TARGET = 518
# (height, width): portrait/landscape phone frames, 4:3, non-multiples of 14, square.
SHAPES = [(1920, 1080), (1080, 1920), (1280, 720), (720, 1280), (1440, 1080),
          (1088, 1920), (1000, 777), (777, 1000), (900, 900), (640, 480)]


def upstream_preprocess(image: np.ndarray, mode: str) -> np.ndarray:
    height, width = image.shape
    if mode == "pad":
        if width >= height:
            new_width = TARGET
            new_height = round(height * (new_width / width) / 14) * 14
        else:
            new_height = TARGET
            new_width = round(width * (new_height / height) / 14) * 14
    else:
        new_width = TARGET
        new_height = round(height * (new_width / width) / 14) * 14
    out = np.asarray(
        Image.fromarray(image).resize((new_width, new_height), Image.Resampling.BICUBIC),
        dtype=float,
    )
    if mode == "crop" and new_height > TARGET:
        start = (new_height - TARGET) // 2
        out = out[start : start + TARGET]
    if mode == "pad":
        h_pad, w_pad = TARGET - out.shape[0], TARGET - out.shape[1]
        if h_pad > 0 or w_pad > 0:
            top, left = h_pad // 2, w_pad // 2
            out = np.pad(out, ((top, h_pad - top), (left, w_pad - left)))
    return out


def centroid(image: np.ndarray) -> tuple[float, float]:
    weights = np.clip(image, 0, None)
    y, x = np.indices(weights.shape)
    return float((weights * x).sum() / weights.sum()), float((weights * y).sum() / weights.sum())


@pytest.mark.parametrize("mode", ["pad", "crop"])
@pytest.mark.parametrize("shape", SHAPES)
def test_mapping_predicts_marker_position_in_upstream_preprocessing(shape, mode):
    height, width = shape
    probes = [(0.5, 0.5), (0.2, 0.3), (0.8, 0.7), (0.5, 0.9)]
    for fx, fy in probes:
        cx, cy = int(fx * (width - 1)), int(fy * (height - 1))
        image = np.zeros(shape, np.uint8)
        # 15 px marker: large enough to survive downscaling, small enough to localise.
        image[cy - 7 : cy + 8, cx - 7 : cx + 8] = 255
        processed = upstream_preprocess(image, mode)
        if processed.max() < 8:  # marker removed by the centre crop; mapping tested elsewhere
            continue
        matrix = native_to_geometry_matrix(shape, processed.shape, mode)
        predicted = (matrix @ np.array([cx, cy, 1.0]))[:2]
        observed = np.array(centroid(processed))
        assert np.abs(predicted - observed).max() < 0.1, (shape, mode, (fx, fy), predicted, observed)


@pytest.mark.parametrize("shape", SHAPES)
def test_pad_mode_returns_square_model_input(shape):
    processed = upstream_preprocess(np.zeros(shape, np.uint8), "pad")
    assert processed.shape == (TARGET, TARGET)
    matrix = native_to_geometry_matrix(shape, processed.shape, "pad")
    assert matrix[0, 0] == pytest.approx(matrix[1, 1], rel=0.02)  # near-isotropic scale
