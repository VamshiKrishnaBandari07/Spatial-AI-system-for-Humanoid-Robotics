import cv2
import numpy as np
from spatialmemory.classical_changes import align, difference, features


def test_unchanged_and_global_brightness_do_not_create_regions():
    im=np.full((240,320,3),100,np.uint8)
    _,_,regions,_=difference(im,im+20,np.eye(3))
    assert regions==[]


def test_local_change_is_saved_as_region_not_physical_class():
    a=np.full((240,320,3),100,np.uint8);b=a.copy();b[80:140,100:160]=(20,20,220)
    _,mask,regions,_=difference(a,b,np.eye(3))
    assert mask[100,120]>0
    assert any(r['bbox_xyxy'][0]<=120<r['bbox_xyxy'][2] for r in regions)
    assert all(r.get('physical_change_class') is None for r in regions)
    assert all(r.get('classification')=='VISUAL_CHANGE_CANDIDATE' for r in regions)


def test_border_hugging_difference_is_rejected():
    a=np.full((240,320,3),100,np.uint8)
    b=a.copy()
    b[:, :12] = (20, 20, 220)
    _,_,regions,stats=difference(a,b,np.eye(3))
    assert stats['rejected_border_regions'] >= 1
    assert regions == []


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
    _,mask,_,_=difference(im,shifted,np.linalg.inv(H))
    assert not mask[:,-15:].any()


def test_html_escape_helper_rejects_injection_chars():
    from spatialmemory.classical_changes import _escape
    assert '&lt;' in _escape('<script>')
    assert '&quot;' in _escape('"x"')
