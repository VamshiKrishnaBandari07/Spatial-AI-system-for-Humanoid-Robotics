import numpy as np
from spatialmemory.camera_refinement import refine_cameras
from spatialmemory.geometry import Reconstruction, project


def synthetic_scene():
    rng = np.random.default_rng(5)
    points = rng.uniform([-1.,-.6,3.], [1.,.6,5.], size=(60,3))
    k = np.array([[500.,0,320],[0,500.,240],[0,0,1]])
    poses, features, tracks = {}, {}, {}
    for i in range(4):
        fid = str(i)
        pose = np.eye(4)
        pose[0,3] = -.3*i
        uv,_ = project(points,k,pose)
        features[fid] = (uv,None,None)
        for j in range(len(points)):
            tracks[(fid,j)] = j
        if i >= 2:
            pose[1,3] += .02
        poses[fid] = pose
    return Reconstruction(points,np.zeros_like(points),poses,tracks,features,k,{})


def test_joint_camera_refinement_improves_withheld_pixels_and_fixes_gauge():
    source = synthetic_scene()
    original = source.poses['2'].copy()
    result = refine_cameras(source,max_nfev=80)
    audit = result.diagnostics['joint_camera_refinement']
    assert audit['accepted']
    assert audit['refined_heldout_median_px'] < audit['original_heldout_median_px']
    assert audit['refined_heldout_inliers'] >= audit['original_heldout_inliers']
    np.testing.assert_array_equal(source.poses['2'],original)
    for fid in ['0','1']:
        np.testing.assert_array_equal(source.poses[fid],result.poses[fid])


def test_insufficient_camera_evidence_retains_original_poses():
    source = synthetic_scene()
    source.tracks = {}
    result = refine_cameras(source)
    assert result is source
    assert not result.diagnostics['joint_camera_refinement']['accepted']
