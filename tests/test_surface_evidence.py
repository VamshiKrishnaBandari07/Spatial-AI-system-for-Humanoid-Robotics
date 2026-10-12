import numpy as np
import pytest
from spatialmemory.surface_evidence import sample_surface
from spatialmemory.semantics import common_frame, ObjectState, visibility_evidence
from spatialmemory.contracts import Config, Registration
from spatialmemory.memory import Observation
from spatialmemory.geometry import Reconstruction
from spatialmemory.visibility import visibility_details


def test_samples_are_observed_points_not_box_corners():
    points = np.arange(300,dtype=float).reshape(100,3)
    sampled = sample_surface(points)
    assert len(sampled)==32
    assert all(any(np.array_equal(point,original) for original in points) for point in sampled)
    with pytest.raises(ValueError):
        sample_surface(np.array([[np.nan,0,0]]))


def test_surface_samples_follow_shared_frame_transform():
    observation = Observation(id='a',frame_id='a',timestamp=0,label='cup',viewpoint='a',
        position=[0,0,1],provenance={'camera_center':[0,0,0],'observed_surface_samples':[[0,0,1]]})
    registration = Registration(valid=True,scale=2,rotation=np.eye(3).tolist(),translation=[1,0,0],
                                 inliers=3,correspondences=3,rmse=0,threshold=.1,reason='test')
    result = common_frame([observation],registration)[0]
    assert result.provenance['observed_surface_samples']==[[1,0,2]]
    assert observation.provenance['observed_surface_samples']==[[0,0,1]]


def test_missing_surface_evidence_cannot_fall_back_to_box():
    obj = ObjectState('a','cup',[0,0,1],None,[],[[0,0,0],[1,1,1]])
    reconstruction = Reconstruction(np.empty((0,3)),np.empty((0,3)),{},{},{},np.eye(3),{})
    result = visibility_evidence(obj,reconstruction,Config(observed_surface_visibility_enabled=True))
    assert result['supported_free_views']==0
    assert result['source']=='insufficient observed surface samples'


def test_support_surface_at_box_boundary_does_not_replace_object_surface():
    k = np.array([[40.,0,32],[0,40.,32],[0,0,1]])
    poses = {'a':np.eye(4),'b':np.eye(4)}
    poses['b'][0,3]=-.8
    surfaces = {fid:{'depth':np.full((64,64),3.),'confidence':np.ones((64,64)),
                     'intrinsics':k} for fid in poses}
    reconstruction = Reconstruction(np.empty((0,3)),np.empty((0,3)),poses,{},{},k,{},surfaces)
    samples = [[x,y,2.] for x in [-.1,.1] for y in [-.1,.1]]
    observation = Observation(id='a',frame_id='a',timestamp=0,label='box',viewpoint='a',
                              provenance={'observed_surface_samples':samples})
    obj = ObjectState('a','box',[0,0,2],None,[observation],[[-.1,-.1,2],[.1,.1,3]])
    assert visibility_evidence(obj,reconstruction,Config())['supported_free_views']==0
    result = visibility_evidence(obj,reconstruction,Config(observed_surface_visibility_enabled=True))
    assert result['supported_free_views']==2
    assert result['sample_count']==4


def test_ray_diagnostics_distinguish_missing_evidence_without_fabricating_free():
    k=np.array([[20.,0,10],[0,20.,10],[0,0,1]])
    depth=np.full((20,20),3.)
    support=np.ones((20,20),bool)
    support[10,10]=False
    points=np.array([[0.,0,2],[100.,0,2],[0,0,-1]])
    rows=visibility_details(points,depth,k,np.eye(4),support)
    assert [r['reason'] for r in rows]==['depth_support_gate_rejected','outside_image','invalid_or_behind_camera']
    assert all(r['state']=='UNKNOWN' for r in rows)
    support[10,10]=True
    assert visibility_details(points[:1],depth,k,np.eye(4),support)[0]['state']=='FREE'
