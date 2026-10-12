import numpy as np
from spatialmemory.frame_comparison import correspondence, frames
import pytest


def test_no_texture_does_not_invent_correspondence():
    frame = {"descriptor": None, "points": np.empty((0, 2))}
    assert correspondence(frame, frame) is None


def test_supported_translation_has_correct_transform_direction():
    rng = np.random.default_rng(9)
    points = rng.uniform(10, 200, (30, 2))
    descriptors = rng.random((30, 128)).astype(np.float32)
    a = {"points": points, "descriptor": descriptors}
    b = {"points": points + [8, -4], "descriptor": descriptors}
    result = correspondence(a, b)
    assert result["inliers"] == 30
    np.testing.assert_allclose(
        np.asarray(result["homography_before_to_after"])[:2, 2], [8, -4], atol=1e-4
    )


def test_invalid_sampling_configuration():
    with pytest.raises(ValueError):
        frames("missing.mp4", -1, 5)
