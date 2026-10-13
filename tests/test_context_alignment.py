import cv2
import numpy as np
import pytest
from spatialmemory.context_alignment import fit_context


def test_context_tracks_surface_when_object_disappears():
    rng=np.random.default_rng(3)
    source=rng.integers(0,256,(400,400,3),dtype=np.uint8)
    source=cv2.GaussianBlur(source,(3,3),0)
    target=cv2.warpAffine(source,np.float32([[1,0,15],[0,1,10]]),(400,400))
    source[160:220,160:220]=[0,0,255]
    result=fit_context(source,target,np.array([160,160,220,220]))
    assert result is not None
    assert np.allclose(result.polygon,[[175,170],[235,170],[235,230],[175,230]],atol=1)
    assert result.heldout_error_px<=2


def test_no_context_does_not_fabricate_mapping():
    blank=np.zeros((300,300,3),np.uint8)
    assert fit_context(blank,blank,np.array([100,100,200,200])) is None
    with pytest.raises(ValueError):fit_context(blank,blank,np.array([200,100,100,200]))
