from fastapi.testclient import TestClient

from spatialmemory import app as runtime


def test_upload_uses_conservative_phone_settings():
    config = runtime.upload_config("vggt", True)
    assert config.neural_frames_per_video == 4
    assert config.neural_frame_selection == "quality_diverse"
    assert config.association_mode == "multiview"
    assert config.movement_mode == "bounded"
    assert config.movement_depth_error_bound is None
    assert config.movement_registration_error_bound is None
    assert config.geometry_backend == "vggt" and config.semantics_enabled


def test_artifact_download_is_job_scoped_and_allowlisted(tmp_path, monkeypatch):
    monkeypatch.setattr(runtime, "ROOT", tmp_path)
    jid = "c" * 32
    folder = tmp_path / jid / "output/reconstruction"
    folder.mkdir(parents=True)
    (folder / "aligned_scene.ply").write_text("ply\n")
    client = TestClient(runtime.app)
    response = client.get(f"/jobs/{jid}/artifacts/aligned")
    assert response.status_code == 200 and response.text == "ply\n"
    assert "aligned_scene.ply" in response.headers["content-disposition"]
    assert client.get(f"/jobs/{jid}/artifacts/unknown").status_code == 404
    assert client.get(f"/jobs/{jid}/artifacts/before").status_code == 404
    assert client.get(f"/jobs/{'d' * 32}/artifacts/aligned").status_code == 404
    outside = tmp_path / "private.ply"
    outside.write_text("private")
    (folder / "before.ply").symlink_to(outside)
    assert client.get(f"/jobs/{jid}/artifacts/before").status_code == 404
