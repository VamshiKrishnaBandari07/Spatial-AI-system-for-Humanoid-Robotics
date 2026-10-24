import json
from fastapi.testclient import TestClient
import spatialmemory.app as service


def test_classical_page_and_scoped_artifacts(tmp_path,monkeypatch):
    monkeypatch.setattr(service,'ROOT',tmp_path)
    client=TestClient(service.app)
    assert client.get('/classical').status_code==200
    jid='a'*32;root=tmp_path/jid/'classical';root.mkdir(parents=True)
    (root/'report.json').write_text(json.dumps({'groups':[]}))
    (tmp_path/jid/'private.json').write_text('{}')
    assert client.get(f'/jobs/{jid}/classical/report.json').status_code==200
    assert client.get(f'/jobs/{jid}/classical/%2e%2e/private.json').status_code==404
    assert client.get(f'/jobs/{jid}').json()['status']=='complete'
