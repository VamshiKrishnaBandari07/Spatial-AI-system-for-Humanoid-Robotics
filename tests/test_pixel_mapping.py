import numpy as np
import pytest
from spatialmemory.pixel_mapping import mask_to_geometry
from spatialmemory.contracts import Config


def test_default_phone_geometry_preserves_objects_at_both_portrait_edges():
    mask = np.zeros((832, 464), bool)
    mask[20:60, 100:150] = True
    mask[740:790, 100:150] = True
    mapped = mask_to_geometry(mask, (518, 518), Config().neural_preprocessing)
    assert mapped[:60].any()
    assert mapped[450:].any()
    assert Config(neural_preprocessing="crop").neural_preprocessing == "crop"


def test_portrait_bottom_object_survives_pad_but_not_crop():
    mask = np.zeros((832, 464), bool)
    mask[740:790, 100:150] = True
    assert not mask_to_geometry(mask, (518, 518), "crop").any()
    padded = mask_to_geometry(mask, (518, 518), "pad")
    assert padded.any()
    assert not padded[:, :112].any()
    assert not padded[:, 406:].any()


def test_mapping_rejects_wrong_model_dimensions():
    with pytest.raises(ValueError, match="dimensions"):
        mask_to_geometry(np.ones((832, 464), bool), (200, 200), "pad")


def test_landscape_native_content_has_correct_padding():
    mask = mask_to_geometry(np.ones((400, 800), bool), (518, 518), "pad")
    assert mask[133:385].all()
    assert not mask[:126].any()
