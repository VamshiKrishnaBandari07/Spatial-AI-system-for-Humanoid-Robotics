"""Regression cases discovered during the testing-console and output review."""

import asyncio
import io
import json

import numpy as np
import pytest
from fastapi import HTTPException, UploadFile
from fastapi.testclient import TestClient
from pydantic import ValidationError

from spatialmemory import app as runtime
from spatialmemory.exports import export_run, ply
from spatialmemory.inputs import Camera, DepthMetadata


def test_changed_regions_excludes_unchanged_and_unknown(tmp_path):
    report = {
        "events": [
            {
                "change": "UNCHANGED",
                "before_position": [99, 99, 99],
                "after_position": [99, 99, 99],
            },
            {
                "change": "UNKNOWN",
                "before_position": [88, 88, 88],
                "after_position": None,
            },
            {
                "change": "MOVED",
                "before_position": [1, 2, 3],
                "after_position": [4, 5, 6],
            },
            {"change": "ADDED", "before_position": None, "after_position": [7, 8, 9]},
            {
                "change": "REMOVED",
                "before_position": [10, 11, 12],
                "after_position": None,
            },
            {"change": "ADDED", "before_position": None, "after_position": None},
        ],
        "registration": {},
        "stages": [],
    }
    empty = np.empty((0, 3))
    export_run(tmp_path, report, empty, empty, empty, empty)
    text = (tmp_path / "changes/changed_regions.ply").read_text()
    assert "element vertex 4\n" in text
    coordinates = np.array(
        [
            list(map(float, line.split()[:3]))
            for line in text.split("end_header\n")[1].splitlines()
        ]
    )
    np.testing.assert_equal(
        coordinates, [[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]]
    )


def test_ply_rejects_truncated_color_rows_and_nonfinite_geometry(tmp_path):
    path = tmp_path / "invalid.ply"
    with pytest.raises(ValueError, match="match"):
        ply(path, np.ones((2, 3)), np.ones((1, 3)))
    assert not path.exists()
    with pytest.raises(ValueError, match="finite"):
        ply(path, [[np.nan, 0, 0]])
    assert not path.exists()


@pytest.mark.parametrize(
    "field,value", [("fx", float("inf")), ("cx", float("nan")), ("fy", float("inf"))]
)
def test_camera_calibration_rejects_nonfinite_values(field, value):
    camera = dict(width=640, height=480, fx=500, fy=500, cx=320, cy=240, source="test")
    camera[field] = value
    with pytest.raises(ValidationError):
        Camera(**camera)


def test_depth_extrinsics_reject_nonrigid_matrix():
    matrix = np.eye(4)
    matrix[0, 0] = 2
    with pytest.raises(ValidationError, match="rigid"):
        DepthMetadata(
            unit="metres",
            width=640,
            height=480,
            camera="camera.json",
            rgb_aligned=False,
            T_rgb_depth=matrix.tolist(),
            frames=[],
        )


def test_empty_upload_closes_both_files_and_releases_active_job(tmp_path, monkeypatch):
    monkeypatch.setattr(runtime, "ROOT", tmp_path)
    monkeypatch.setattr(runtime, "JOBS", {})
    monkeypatch.setattr(
        runtime,
        "neural_capabilities",
        lambda: {"sparse": True, "vggt": False, "semantics": False},
    )
    before = UploadFile(filename="before.mp4", file=io.BytesIO())
    after = UploadFile(filename="after.mp4", file=io.BytesIO(b"video"))
    with pytest.raises(HTTPException) as failure:
        asyncio.run(runtime.submit(before, after, "sparse", False))
    assert failure.value.status_code == 400
    assert before.file.closed and after.file.closed
    assert all(state["status"] == "failed" for state in runtime.JOBS.values())


def test_job_directory_failure_does_not_leave_queue_blocked(tmp_path, monkeypatch):
    not_directory = tmp_path / "file"
    not_directory.write_text("preserve")
    monkeypatch.setattr(runtime, "ROOT", not_directory)
    monkeypatch.setattr(runtime, "JOBS", {})
    monkeypatch.setattr(
        runtime,
        "neural_capabilities",
        lambda: {"sparse": True, "vggt": False, "semantics": False},
    )
    before = UploadFile(filename="before.mp4", file=io.BytesIO(b"video"))
    after = UploadFile(filename="after.mp4", file=io.BytesIO(b"video"))
    with pytest.raises(OSError):
        asyncio.run(runtime.submit(before, after, "sparse", False))
    assert all(state["status"] == "failed" for state in runtime.JOBS.values())
    assert before.file.closed and after.file.closed
    assert not_directory.read_text() == "preserve"


def test_worker_rejection_is_recoverable_and_persisted(tmp_path, monkeypatch):
    class RejectWorker:
        def submit(self, function):
            raise RuntimeError("Executor unavailable")

    class AcceptWorker:
        def submit(self, function):
            pass  # Queue only; regression does not run or mock model predictions.

    monkeypatch.setattr(runtime, "ROOT", tmp_path)
    monkeypatch.setattr(runtime, "JOBS", {})
    monkeypatch.setattr(
        runtime,
        "neural_capabilities",
        lambda: {"sparse": True, "vggt": False, "semantics": False},
    )
    monkeypatch.setattr(runtime, "POOL", RejectWorker())
    client = TestClient(runtime.app)
    files = {"before": ("before.mp4", b"video"), "after": ("after.mp4", b"video")}
    response = client.post("/jobs", files=files)
    assert response.status_code == 503
    job = next(tmp_path.iterdir())
    assert json.loads((job / "job.json").read_text())["status"] == "failed"
    assert client.get("/jobs/" + job.name).json()["status"] == "failed"
    monkeypatch.setattr(runtime, "POOL", AcceptWorker())
    assert client.post("/jobs", files=files).status_code == 200


def test_cancelled_upload_cleans_partial_file_and_releases_queue(tmp_path, monkeypatch):
    monkeypatch.setattr(runtime, "ROOT", tmp_path)
    monkeypatch.setattr(runtime, "JOBS", {})
    monkeypatch.setattr(
        runtime,
        "neural_capabilities",
        lambda: {"sparse": True, "vggt": False, "semantics": False},
    )
    before = UploadFile(filename="before.mp4", file=io.BytesIO(b"partial"))
    after = UploadFile(filename="after.mp4", file=io.BytesIO(b"video"))

    async def interrupted_read(size):
        raise asyncio.CancelledError()

    monkeypatch.setattr(before, "read", interrupted_read)
    with pytest.raises(asyncio.CancelledError):
        asyncio.run(runtime.submit(before, after, "sparse", False))
    assert before.file.closed and after.file.closed
    assert all(state["status"] == "failed" for state in runtime.JOBS.values())
    assert not list(tmp_path.rglob("before.mp4"))


def test_oversize_upload_rejected_and_queue_released(tmp_path, monkeypatch):
    monkeypatch.setattr(runtime, "ROOT", tmp_path)
    monkeypatch.setattr(runtime, "JOBS", {})
    monkeypatch.setattr(runtime, "MAX_UPLOAD", 3)
    monkeypatch.setattr(
        runtime,
        "neural_capabilities",
        lambda: {"sparse": True, "vggt": False, "semantics": False},
    )
    response = TestClient(runtime.app).post(
        "/jobs",
        files={"before": ("before.mp4", b"1234"), "after": ("after.mp4", b"12")},
    )
    assert response.status_code == 413
    assert all(state["status"] == "failed" for state in runtime.JOBS.values())
    assert not list(tmp_path.rglob("before.mp4"))
