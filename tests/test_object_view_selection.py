import numpy as np
from spatialmemory.object_view_selection import object_view_coverage,rank_object_views


def test_object_coverage_rejects_wrong_view_not_empty_depth():
    points=np.array([[-.2,-.2,2],[.2,.2,2],[0,0,2]])
    k=np.array([[20,0,10],[0,20,10],[0,0,1]])
    good=object_view_coverage(points,k,np.eye(4),(20,20))
    wrong=np.eye(4);wrong[0,3]=100
    bad=object_view_coverage(points,k,wrong,(20,20))
    assert good['in_frame_fraction']==1 and bad['in_frame_fraction']==0
    assert not good['occupancy_used_for_selection']
    assert rank_object_views([{'frame_id':'bad','object_projection':bad},{'frame_id':'good','object_projection':good}])==['good']


def test_padding_and_behind_camera_count_against_coverage():
    k=np.array([[20,0,10],[0,20,10],[0,0,1]])
    points=np.array([[0,0,2],[0,0,-2]])
    r=object_view_coverage(points,k,np.eye(4),(20,20))
    assert r['in_frame_fraction']==.5
    mask=np.ones((20,20),bool);mask[10,10]=False
    assert object_view_coverage(points,k,np.eye(4),(20,20),mask)['in_frame_fraction']==0
