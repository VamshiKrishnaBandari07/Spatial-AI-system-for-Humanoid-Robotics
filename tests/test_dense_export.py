import numpy as np
import pytest
from spatialmemory.dense_export import transform_after


def test_explicit_after_to_before_sim3():
    r={'valid':True,'scale':2,'rotation':[[0,-1,0],[1,0,0],[0,0,1]],'translation':[3,4,5]}
    np.testing.assert_allclose(transform_after(np.array([[1,0,1]]),r),[[3,6,7]])
    r['valid']=False
    with pytest.raises(ValueError):transform_after(np.zeros((1,3)),r)
    r['valid']=True;r['scale']=-1
    with pytest.raises(ValueError):transform_after(np.zeros((1,3)),r)


def test_sensor_registration_cannot_transform_unscaled_neural_points(tmp_path):
    import json
    from spatialmemory.dense_export import export_dense
    (tmp_path/'report.json').write_text(json.dumps({'status':'complete','config':{
        'neural_frame_selection':'quality_diverse','geometry_backend':'vggt',
        'metric_geometry_enabled':True}}))
    with pytest.raises(ValueError,match='sensor replacement'):
        export_dense(tmp_path,tmp_path/'out')
    assert not (tmp_path/'out').exists()
