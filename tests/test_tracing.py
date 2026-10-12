import json
import numpy as np
import pytest
from spatialmemory.tracing import PipelineTracer, export_run_trace


def test_empty_trace_is_not_run(capsys):
    tracer = PipelineTracer()
    tracer.summarize_failures()
    assert 'NOT_RUN' in capsys.readouterr().out


def test_full_mask_inside_box_not_rejected_and_leakage_separate(tmp_path):
    tracer = PipelineTracer()
    mask = np.zeros((20, 20), np.uint8)
    mask[5:15, 5:15] = 1
    row = tracer.audit_crop_and_mask('object', 12, [5,5,15,15], mask, .45,
                                      np.zeros((20,20,3), np.uint8))
    assert row.status == 'WARN'
    assert row.metrics['mask_coverage'] == 1
    assert row.metrics['mask_leakage'] == 0
    path = tracer.save(tmp_path)
    saved = json.loads(path.read_text())
    assert saved['steps'][0]['artifact_path']
    assert (tmp_path / saved['steps'][0]['artifact_path']).exists()


@pytest.mark.parametrize('error', [2., 4.86, float('nan'), float('inf'), -1])
def test_invalid_or_boundary_geometry_fails(error):
    assert PipelineTracer().audit_3d_localization('object', 12, error).status == 'FAIL'


def test_visibility_count_and_score_do_not_prove_empty_surface():
    tracer = PipelineTracer()
    assert tracer.audit_visibility_and_removal('object', 12, 2, .99).status == 'FAIL'
    assert tracer.audit_visibility_and_removal('object', 12, 2, .99,
             verified_view_ids=['a','a'], patches_verified=True).status == 'FAIL'
    assert tracer.audit_visibility_and_removal('object', 12, 2, .99,
             verified_view_ids=['a','b'], patches_verified=True).status == 'PASS'


def test_identity_requires_valid_embedding_and_competitor_margin():
    tracer = PipelineTracer()
    assert tracer.audit_features_and_matching('object', 0, np.zeros(3), .5).status == 'FAIL'
    assert tracer.audit_features_and_matching('object', 0, np.ones(3), .2, .19).status == 'FAIL'
    assert tracer.audit_features_and_matching('object', 0, np.ones(3), None).status == 'WARN'


def test_saved_run_export_uses_native_evidence_directory(tmp_path):
    import cv2
    from spatialmemory.memory import Memory, Observation
    evidence = tmp_path / 'evidence/supporting_frames'
    evidence.mkdir(parents=True)
    mask = np.zeros((20,20), np.uint8)
    mask[5:15,5:15] = 255
    cv2.imwrite(str(evidence / 'mask.png'), mask)
    image = tmp_path / 'frame.jpg'
    cv2.imwrite(str(image), np.zeros((20,20,3), np.uint8))
    before = tmp_path / 'before'
    before.mkdir()
    (before / 'manifest.json').write_text(json.dumps({'frames':[{'id':'before_12','path':str(image)}]}))
    memory = Memory(tmp_path / 'spatial_memory.sqlite')
    memory.observe(Observation(id='object_12',frame_id='before_12', timestamp=1., label='cup',
                   viewpoint='before_12', provenance={'mask_file':'mask.png','box':[5,5,15,15],'raw_detector_score':.8}))
    memory.close()
    exported = json.loads(export_run_trace(tmp_path).read_text())
    assert exported['steps'][0]['status'] == 'PASS'
    assert exported['steps'][0]['artifact_path']
    assert exported['summary']['status'] == 'WARN'
