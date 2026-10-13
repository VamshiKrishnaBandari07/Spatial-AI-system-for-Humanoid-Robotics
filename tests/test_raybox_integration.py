import numpy as np
import pytest
from types import SimpleNamespace
from spatialmemory.raybox_integration import conformal_depth_bound, propagated_geometry_bound, raybox_view


def test_conformal_finite_sample_rank():
    assert conformal_depth_bound(np.arange(1, 20), .1) == 18
    with pytest.raises(ValueError, match='Insufficient'):
        conformal_depth_bound(np.ones(3), .05)


def test_geometry_translation_and_scale_bounds():
    corners = np.array([[0, 0, 2], [0, 0, 4]])
    bound = propagated_geometry_bound(corners, np.eye(4), np.eye(4),
        localization_radius=.1, scale_error=.1, registration_angle=0,
        registration_translation=.2, camera_angle=0, camera_translation=.3)
    assert bound == pytest.approx(1.01)


def test_actual_adapter_inverts_registration_and_missing_bounds_blocks():
    surface = {'depth': np.full((5,5), 10.), 'intrinsics': np.array([[4,0,2],[0,4,2],[0,0,1]])}
    reg = SimpleNamespace(valid=True, scale=2., rotation=np.eye(3), translation=[0,0,2])
    extent = [[-1,-1,6],[1,1,10]]
    args = (extent, surface, np.eye(4), np.ones((5,5),bool),reg)
    r=raybox_view(*args)
    assert r['reason']=='missing_validated_uncertainty_bounds'
    surface['raybox_uncertainty']={'depth_error_bound':.1,'geometry_error_bound':.1,'scale_id':'after','source':'synthetic test only'}
    r=raybox_view(*args)
    assert r['status']=='diagnostic_only'
    assert r['counts']['FREE']>0
    assert not r['promotes_change_decision']
    assert np.asarray(r['object_to_target_world'])[2,3] == -1
