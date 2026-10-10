import json

import numpy as np
from fastapi.testclient import TestClient

from spatialmemory import app as runtime


def test_cpu_install_advertises_available_workflow(monkeypatch):
    monkeypatch.setattr(runtime, "find_spec", lambda name: None)
    health = TestClient(runtime.app).get("/health").json()
    assert health["capabilities"] == {"sparse": True, "vggt": False, "semantics": False}


def test_unavailable_neural_profile_rejected_before_upload(tmp_path, monkeypatch):
    monkeypatch.setattr(runtime, "ROOT", tmp_path)
    monkeypatch.setattr(runtime, "find_spec", lambda name: None)
    response = TestClient(runtime.app).post(
        "/jobs",
        data={"geometry_backend": "vggt"},
        files={"before": ("before.mp4", b"video"), "after": ("after.mp4", b"video")},
    )
    assert response.status_code == 503
    assert "not installed" in response.json()["detail"]
    assert list(tmp_path.iterdir()) == []


def test_failed_job_survives_restart(tmp_path, monkeypatch):
    monkeypatch.setattr(runtime, "ROOT", tmp_path)
    monkeypatch.setattr(runtime, "JOBS", {})
    job = tmp_path / ("a" * 32)
    job.mkdir()
    state = {"status": "failed", "stage": "registration", "error": "No overlap"}
    runtime.persist_status(job, state)
    client = TestClient(runtime.app)
    assert client.get("/jobs/" + job.name).json() == state
    assert not (job / "job.json.tmp").exists()
    (job / "job.json").unlink()
    (job / "output").mkdir()
    (job / "output/failure.json").write_text(json.dumps(state))
    assert client.get("/jobs/" + job.name).json() == state


def test_world_response_enforces_point_budget(tmp_path, monkeypatch):
    monkeypatch.setattr(runtime, "ROOT", tmp_path)
    job = tmp_path / ("b" * 32)
    (job / "output").mkdir(parents=True)
    points = np.arange(29999 * 3).reshape(-1, 3)
    colors = points % 256
    np.savez(job / "output/world.npz", points=points, colors=colors)
    data = TestClient(runtime.app).get("/jobs/" + job.name + "/world").json()
    assert len(data["points"]) <= 15000
    assert len(data["points"]) == len(data["colors"])
    assert data["points"][1] == points[2].tolist()
    assert data["colors"][1] == colors[2].tolist()


def test_evaluation_uses_ids_and_reports_unannotated_objects(tmp_path):
    import subprocess
    import sys

    from spatialmemory.inputs import Manifest, Recording

    manifest = Manifest(
        scene_id="s",
        before=Recording(video="before.mp4"),
        after=Recording(video="after.mp4"),
    )
    path = tmp_path / "manifest.json"
    path.write_text(manifest.model_dump_json())
    annotations = tmp_path / "annotations.json"
    annotations.write_text(
        json.dumps(
            {
                "schema_version": "1.0",
                "scene_id": "s",
                "coordinate_frame": "shared_world",
                "position_unit": "unknown",
                "scale_verified": False,
                "changes": [
                    {
                        "object_id": "chair_1",
                        "label": "chair",
                        "change_type": "UNCHANGED",
                        "visibility": "OBSERVED",
                        "annotation_source": "synthetic test",
                    }
                ],
            }
        )
    )
    report = tmp_path / "report.json"
    report.write_text(
        json.dumps(
            {
                "scene_id": "s",
                "events": [
                    {"object_id": "chair_1", "change": "UNCHANGED"},
                    {"object_id": "local_2", "change": "UNKNOWN"},
                ],
            }
        )
    )
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "spatialmemory_x",
            "evaluate",
            "--manifest",
            str(path),
            "--annotations",
            str(annotations),
            "--report",
            str(report),
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    data = json.loads(result.stdout)
    assert data["accuracy"] == 1.0 and data["sample_count"] == 1
    assert data["unannotated_object_predictions"] == ["local_2"]
