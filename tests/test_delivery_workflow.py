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


def test_result_bundle_preserves_outputs_and_excludes_private_links(
    tmp_path, monkeypatch
):
    import io
    import zipfile

    monkeypatch.setattr(runtime, "ROOT", tmp_path)
    jid = "e" * 32
    output = tmp_path / jid / "output"
    (output / "reconstruction").mkdir(parents=True)
    (output / "report.json").write_text('{"status":"complete"}')
    (output / "reconstruction/before.ply").write_text("ply\n")
    (output.parent / "before.mp4").write_bytes(b"raw-private-recording")
    private = tmp_path / "private.json"
    private.write_text('"private"')
    (output / "reconstruction/private.json").symlink_to(private)
    client = TestClient(runtime.app)
    response = client.get(f"/jobs/{jid}/results")
    assert response.status_code == 200
    with zipfile.ZipFile(io.BytesIO(response.content)) as archive:
        assert set(archive.namelist()) == {"report.json", "reconstruction/before.ply"}
        assert archive.read("reconstruction/before.ply") == b"ply\n"
        assert archive.testzip() is None
    assert (output / "reconstruction/before.ply").read_text() == "ply\n"
    (output / "report.json").unlink()
    assert client.get(f"/jobs/{jid}/results").status_code == 409


def test_second_upload_failure_removes_only_incomplete_pair(tmp_path, monkeypatch):
    import asyncio
    import io
    import pytest
    from fastapi import HTTPException, UploadFile

    monkeypatch.setattr(runtime, "ROOT", tmp_path)
    monkeypatch.setattr(runtime, "JOBS", {})
    monkeypatch.setattr(
        runtime,
        "neural_capabilities",
        lambda: {"sparse": True, "vggt": False, "semantics": False},
    )
    previous = tmp_path / "previous-result.json"
    previous.write_text('"preserve"')
    before = UploadFile(filename="before.mp4", file=io.BytesIO(b"video"))
    after = UploadFile(filename="after.mp4", file=io.BytesIO())
    with pytest.raises(HTTPException):
        asyncio.run(runtime.submit(before, after, "sparse", False))
    assert not list(tmp_path.rglob("*.mp4"))
    assert previous.read_text() == '"preserve"'
    assert before.file.closed and after.file.closed
    assert all(state["status"] == "failed" for state in runtime.JOBS.values())
