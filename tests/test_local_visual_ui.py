import json
from concurrent.futures import Future
from fastapi.testclient import TestClient
from spatialmemory import app as runtime
from spatialmemory.local_visual import compare_local
import pytest


def setup(tmp_path, monkeypatch):
    monkeypatch.setattr(runtime, 'ROOT', tmp_path)
    monkeypatch.setattr(runtime, 'VISUAL_JOBS', {})
    jid = 'a' * 32
    path = tmp_path / jid / 'output'
    path.mkdir(parents=True)
    (path / 'report.json').write_text(json.dumps({'status': 'complete'}))
    return TestClient(runtime.app), jid, path


def test_local_disabled_explicitly(tmp_path, monkeypatch):
    client, jid, _ = setup(tmp_path, monkeypatch)
    monkeypatch.delenv('SPATIALMEMORY_VISUAL_BACKEND', raising=False)
    assert client.get(f'/jobs/{jid}/visual-comparison').json()['status'] == 'unavailable'
    assert client.post(f'/jobs/{jid}/visual-comparison').status_code == 503


def test_local_worker_and_no_duplicate_request(tmp_path, monkeypatch):
    from spatialmemory import local_visual
    client, jid, _ = setup(tmp_path, monkeypatch)
    monkeypatch.setenv('SPATIALMEMORY_VISUAL_BACKEND', 'local')
    monkeypatch.delenv('OPENAI_API_KEY', raising=False)
    def fake(run, output):
        output.mkdir(parents=True)
        (output / 'visual_changes.json').write_text(json.dumps({'narrative': 'mock, not inference', 'changes': []}))
    class InlinePool:
        def submit(self, fn):
            fn()
            return Future()
    monkeypatch.setattr(runtime, 'VISUAL_POOL', InlinePool())
    monkeypatch.setattr(local_visual, 'compare_local', fake)
    assert client.post(f'/jobs/{jid}/visual-comparison').status_code == 202
    result = client.get(f'/jobs/{jid}/visual-comparison').json()
    assert result['status'] == 'complete'
    assert result['result']['narrative'] == 'mock, not inference'
    assert client.post(f'/jobs/{jid}/visual-comparison').status_code == 409


def test_unfinished_run_rejected(tmp_path, monkeypatch):
    client, jid, path = setup(tmp_path, monkeypatch)
    monkeypatch.setenv('SPATIALMEMORY_VISUAL_BACKEND', 'local')
    (path / 'report.json').unlink()
    assert client.post(f'/jobs/{jid}/visual-comparison').status_code == 409


def test_local_resource_bounds(tmp_path):
    with pytest.raises(ValueError, match='bounds'):
        compare_local(tmp_path, tmp_path / 'out', frames_per_video=100)
    assert not (tmp_path / 'out').exists()


def test_no_correspondences_preserves_explicit_fallback(tmp_path):
    import cv2
    import numpy as np
    from spatialmemory.local_visual import matched_evidence
    for role in ['before', 'after']:
        folder = tmp_path / role
        folder.mkdir()
        rows = []
        for index in range(2):
            path = folder / f'{index}.jpg'
            cv2.imwrite(str(path), np.zeros((64, 64, 3), np.uint8))
            rows.append({'id': f'{role}:{index}', 'timestamp': index, 'path': str(path)})
        (folder / 'manifest.json').write_text(json.dumps({'frames': rows}))
    with pytest.raises(ValueError, match='Insufficient shared-view coverage'):
        matched_evidence(tmp_path, 2)


def test_overlap_rejects_offscreen_singular_and_projective_poles():
    import numpy as np
    from spatialmemory.local_visual import image_overlap
    assert image_overlap(np.eye(3), (100, 100), (100, 100)) == 1.0
    shift = np.eye(3); shift[0, 2] = 75
    assert image_overlap(shift, (100, 100), (100, 100)) < 0.3
    assert image_overlap(np.zeros((3, 3)), (100, 100), (100, 100)) == 0.0
    pole = np.eye(3); pole[2] = [0.02, 0, -1]
    assert image_overlap(pole, (100, 100), (100, 100)) == 0.0
