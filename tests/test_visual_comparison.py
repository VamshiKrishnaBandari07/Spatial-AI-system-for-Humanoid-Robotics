import json
import pytest
from spatialmemory.visual_comparison import prepare_request, validate_response, compare_run


def fixture_run(root):
    for role in ['before', 'after']:
        directory = root / role
        directory.mkdir(parents=True)
        rows = []
        for i in [2, 0, 1]:
            path = directory / f'{i}.jpg'
            path.write_bytes(b'\xff\xd8\xff' + bytes([i]))
            rows.append({'id': f'{role}:{i}', 'timestamp': i / 3, 'path': str(path)})
        (directory / 'manifest.json').write_text(json.dumps({'frames': rows}))
    return root


def test_sampling_roles_timestamps_and_no_answer_key(tmp_path):
    payload, evidence = prepare_request(fixture_run(tmp_path), 'vision-model', 2)
    assert [r['id'] for r in evidence] == ['before:0', 'before:2', 'after:0', 'after:2']
    assert all(len(r['sha256']) == 64 for r in evidence)
    assert 'perfume' not in json.dumps(payload)
    assert 'jug' not in json.dumps(payload)


def test_outside_frame_rejected(tmp_path):
    run = fixture_run(tmp_path)
    manifest = run / 'before/manifest.json'
    data = json.loads(manifest.read_text())
    data['frames'][1]['path'] = str(tmp_path / 'outside.jpg')
    manifest.write_text(json.dumps(data))
    with pytest.raises(ValueError, match='escapes'):
        prepare_request(run, 'vision-model', 2)


def test_invented_and_wrong_role_evidence_rejected():
    evidence = [{'id': 'before:0', 'role': 'before'}, {'id': 'after:0', 'role': 'after'}]
    item = dict(label='container', change='REMOVED', before_frames=['before:0'],
                after_frames=['after:0'], observation='Visible gap',
                visibility_evidence='Support visible', uncertainty='Identity uncertain')
    assert validate_response(json.dumps({'changes': [item], 'limitations': '2D only'}), evidence)
    item['after_frames'] = ['before:0']
    with pytest.raises(ValueError, match='unsupplied'):
        validate_response(json.dumps({'changes': [item], 'limitations': '2D only'}), evidence)


def test_missing_key_fails_without_fabricated_result(tmp_path, monkeypatch):
    monkeypatch.delenv('OPENAI_API_KEY', raising=False)
    run = fixture_run(tmp_path / 'run')
    out = tmp_path / 'results'
    with pytest.raises(RuntimeError, match='no inference'):
        compare_run(run, out, 'vision-model')
    assert (out / 'failure.json').exists()
    assert not (out / 'visual_changes.json').exists()


def test_prepare_is_offline_and_does_not_overwrite(tmp_path):
    run = fixture_run(tmp_path / 'run')
    out = tmp_path / 'results'
    report = compare_run(run, out, 'vision-model', prepare_only=True)
    assert report['status'] == 'prepared_not_inferred'
    assert not report['physical_3d_verified']
    with pytest.raises(FileExistsError):
        compare_run(run, out, 'vision-model', prepare_only=True)
