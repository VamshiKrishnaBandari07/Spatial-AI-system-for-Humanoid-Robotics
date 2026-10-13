import numpy as np
from spatialmemory.crop_recheck import native_region,match_candidate
from spatialmemory.pixel_mapping import native_to_geometry_matrix


def test_native_mapping_round_trip_and_padding():
    a=native_to_geometry_matrix((832,464),(518,518),'pad')
    pts=np.array([[100,200,1],[200,300,1]])@a.T
    box=native_region(pts[:,:2].ravel(),(832,464),(518,518),'pad',context=1)
    assert np.allclose(box,[100,200,200,300],atol=1)
    assert native_region([0,0,20,20],(832,464),(518,518),'pad') is None


def test_lookalikes_and_missing_detections_do_not_become_matches():
    assert match_candidate([]) is None
    assert match_candidate([.91,.90]) is None
    assert match_candidate([.80]) is None
    assert match_candidate([.93,.5]) == 0
