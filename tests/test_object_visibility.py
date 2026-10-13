"""Synthetic mechanics tests; not real-video accuracy evaluation."""
import numpy as np
import pytest
from spatialmemory.object_visibility import evaluate_box_view, EvidenceState as S


def inputs():
    return dict(before_box_min=[-1, -1, 2], before_box_max=[1, 1, 4],
                T_after_before=np.eye(4), T_camera_after=np.eye(4),
                K=np.array([[4., 0, 2], [0, 4, 2], [0, 0, 1]]),
                depth=np.full((5, 5), 5.), depth_valid=np.ones((5, 5), bool),
                depth_error_bound=np.zeros((5, 5)), geometry_error_bound=0.,
                geometry_scale_id='scene_after', depth_scale_id='scene_after')


def center(result):
    return np.flatnonzero(np.all(result.pixels_uv == [2, 2], axis=1))[0]


@pytest.mark.parametrize('depth,state', [(5., S.FREE), (3., S.OCCUPIED), (1., S.OCCLUDED), (2., S.INSUFFICIENT), (4., S.INSUFFICIENT)])
def test_known_camera_z_intervals(depth, state):
    args = inputs(); args['depth'][:] = depth
    result = evaluate_box_view(**args); i = center(result)
    assert result.near_z[i] == 2 and result.far_z[i] == 4
    assert result.states[i] == state
    assert sum(result.fractions.values()) == pytest.approx(1.)


def test_translation_and_rotated_camera():
    args = inputs()
    # BEFORE box along x, AFTER world shifted +10 x, camera looks along +x.
    args['before_box_min'] = [2, -1, -1]; args['before_box_max'] = [4, 1, 1]
    args['T_after_before'][0, 3] = 10
    r = np.array([[0., 0, -1], [0, 1, 0], [1, 0, 0]])
    args['T_camera_after'][:3, :3] = r
    args['T_camera_after'][:3, 3] = -r @ np.array([10, 0, 0])
    result = evaluate_box_view(**args); i = center(result)
    assert (result.near_z[i], result.far_z[i]) == pytest.approx((2, 4))


def test_sim3_scale_rotation_translation():
    args = inputs(); r = np.array([[0., 0, 1], [0, 1, 0], [-1, 0, 0]])
    t = np.array([3., 4, 5])
    args['T_after_before'][:3, :3] = 2 * r
    args['T_after_before'][:3, 3] = t
    args['T_camera_after'][:3, :3] = r.T
    args['T_camera_after'][:3, 3] = -r.T @ t
    args['depth'][:] = 10
    result = evaluate_box_view(**args); i = center(result)
    assert (result.near_z[i], result.far_z[i]) == pytest.approx((4, 8))
    assert result.states[i] == S.FREE


def test_parallel_ray_outside_slab():
    args = inputs(); args['before_box_min'] = [1, -1, 2]; args['before_box_max'] = [3, 1, 4]
    result = evaluate_box_view(**args)
    assert not result.intersects[center(result)]


def test_camera_inside_and_near_plane_crossing():
    args = inputs(); args['before_box_min'] = [-1, -1, -1]; args['before_box_max'] = [1, 1, 1]
    args['near_clip'] = 0.01
    result = evaluate_box_view(**args); i = center(result)
    assert result.near_z[i] == 0.01 and result.far_z[i] == 1
    assert result.touches_image_boundary


def test_center_outside_does_not_discard_partial_box():
    args = inputs(); args['before_box_min'] = [0.5, -1, 2]; args['before_box_max'] = [10, 1, 4]
    result = evaluate_box_view(**args)
    assert result.intersects.any() and result.touches_image_boundary
    assert result.projected_region is not None


def test_behind_camera_has_no_forward_intersections():
    args = inputs(); args['before_box_min'] = [-1, -1, -4]; args['before_box_max'] = [1, 1, -2]
    result = evaluate_box_view(**args)
    assert not result.intersects.any() and result.projected_region is None
    assert all(v == 0 for v in result.counts.values())


@pytest.mark.parametrize('depth,error,geometry', [(np.nan, 0, 0), (np.inf, 0, 0), (0, 0, 0), (4.1, .2, 0), (4.1, 0, .2), (2.1, 0, .2), (3, 0, 2)])
def test_invalid_and_boundary_uncertainty(depth, error, geometry):
    args = inputs(); args['depth'][:] = depth; args['depth_error_bound'][:] = error; args['geometry_error_bound'] = geometry
    result = evaluate_box_view(**args)
    assert result.states[center(result)] == S.INSUFFICIENT


def test_partial_occlusion_and_invalid_depth_counts():
    args = inputs(); args['depth'][2, 2] = 1; args['depth_valid'][2, 3] = False
    result = evaluate_box_view(**args)
    assert result.counts['OCCLUDED'] >= 1
    assert result.counts['FREE'] >= 1
    assert result.counts['INSUFFICIENT'] >= 1


@pytest.mark.parametrize('field,value', [('depth_convention', 'ray_distance'), ('depth_scale_id', 'other'), ('geometry_scale_id', ''), ('pixel_stride', 0), ('geometry_error_bound', -1), ('near_clip', 0), ('before_box_min', [2, 2, 5])])
def test_invalid_metadata_and_bounds(field, value):
    args = inputs(); args[field] = value
    with pytest.raises(ValueError): evaluate_box_view(**args)


@pytest.mark.parametrize('kind', ['reflection', 'shear', 'nonuniform', 'scaled_camera', 'bad_row', 'nan'])
def test_invalid_transforms(kind):
    args = inputs()
    if kind == 'reflection': args['T_after_before'][0, 0] = -1
    elif kind == 'shear': args['T_after_before'][0, 1] = .3
    elif kind == 'nonuniform': args['T_after_before'][0, 0] = 2
    elif kind == 'scaled_camera': args['T_camera_after'][:3, :3] *= 2
    elif kind == 'bad_row': args['T_after_before'][3, 0] = 1
    else: args['T_after_before'][0, 0] = np.nan
    with pytest.raises(ValueError): evaluate_box_view(**args)


def test_shape_and_intrinsics_validation():
    args = inputs(); args['depth_valid'] = np.ones((5, 5))
    with pytest.raises(ValueError): evaluate_box_view(**args)
    args = inputs(); args['K'][0, 0] = 0
    with pytest.raises(ValueError): evaluate_box_view(**args)
    args = inputs(); args['depth_error_bound'][0, 0] = np.nan
    with pytest.raises(ValueError): evaluate_box_view(**args)
