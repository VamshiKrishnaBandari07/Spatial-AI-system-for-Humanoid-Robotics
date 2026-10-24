import cv2
import numpy as np
from spatialmemory.classical_changes import align, difference, features


def test_unchanged_and_global_brightness_do_not_create_regions():
    im=np.full((240,320,3),100,np.uint8)
    _,_,regions=difference(im,im+20,np.eye(3))
    assert regions==[]


def test_local_change_is_saved_as_region_not_physical_class():
    a=np.full((240,320,3),100,np.uint8);b=a.copy();b[80:140,100:160]=(20,20,220)
    _,mask,regions=difference(a,b,np.eye(3))
    assert mask[100,120]>0
    assert any(r['bbox_xyxy'][0]<=120<r['bbox_xyxy'][2] for r in regions)


def test_alignment_rejects_blank_images():
    im=np.zeros((240,320,3),np.uint8)
    assert align(features(im),features(im),im.shape) is None


def test_known_translation_alignment_and_invalid_borders():
    im=np.random.default_rng(3).integers(0,256,(320,400,3),dtype=np.uint8)
    H=np.array([[1.,0,15],[0,1,10],[0,0,1]])
    shifted=cv2.warpPerspective(im,H,(400,320))
    r=align(features(im),features(shifted),im.shape)
    assert r is not None
    assert np.allclose(r[0][:2,2],[-15,-10],atol=.5)
    _,mask,_=difference(im,shifted,np.linalg.inv(H))
    assert not mask[:,-15:].any()
