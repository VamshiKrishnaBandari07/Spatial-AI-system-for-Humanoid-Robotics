import numpy as np
import pytest
from spatialmemory.learned_matching import prepare_crop, LearnedCropMatcher


def test_native_crop_pixel_mapping_roundtrips():
    gray = np.zeros((120,80), np.uint8)
    mask = gray.copy()
    mask[35:55,20:35] = 255
    image, mapping = prepare_crop(gray,mask)
    assert image.shape[:2] == (1,1)
    assert image.shape[-1] % 8 == 0 and image.shape[-2] % 8 == 0
    native = np.array([25.,45.,1.])
    modeled = np.linalg.inv(mapping) @ native
    np.testing.assert_allclose(mapping @ modeled,native)
    assert 0 <= modeled[0] < image.shape[-1]
    assert 0 <= modeled[1] < image.shape[-2]


def test_empty_or_misaligned_masks_rejected():
    with pytest.raises(ValueError,match='empty'):
        prepare_crop(np.zeros((10,10),np.uint8),np.zeros((10,10),np.uint8))
    with pytest.raises(ValueError,match='same-sized'):
        prepare_crop(np.zeros((10,10),np.uint8),np.ones((9,10),np.uint8))


def test_unverified_neural_weights_are_not_loaded(tmp_path):
    path = tmp_path / 'bad.ckpt'
    path.write_bytes(b'not verified weights')
    with pytest.raises(ValueError,match='checksum'):
        LearnedCropMatcher(path)
