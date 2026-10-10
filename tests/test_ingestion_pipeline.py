import importlib.util
from pathlib import Path
import pytest
from spatialmemory.ingestion import ingest
from spatialmemory.contracts import Config
from spatialmemory.pipeline import run

spec = importlib.util.spec_from_file_location(
    "fixture", Path(__file__).parents[1] / "scripts/synthetic_fixture.py"
)
fixture = importlib.util.module_from_spec(spec)
spec.loader.exec_module(fixture)


def test_ingestion_integrity(tmp_path):
    path = fixture.make_video(tmp_path / "input.mp4")
    original = path.read_bytes()
    frames, manifest = ingest(
        path, tmp_path / "frames", Config(sample_seconds=0.2), "before"
    )
    assert manifest["decoded_frames"] == 18
    assert all(a.timestamp < b.timestamp for a, b in zip(frames, frames[1:]))
    assert path.read_bytes() == original and all(Path(f.path).exists() for f in frames)


def test_corrupt_video(tmp_path):
    path = tmp_path / "bad.mp4"
    path.write_bytes(b"not video")
    with pytest.raises(Exception):
        ingest(path, tmp_path / "frames", Config(), "before")


def test_synthetic_video_pipeline(tmp_path):
    b = fixture.make_video(tmp_path / "before.mp4")
    a = fixture.make_video(tmp_path / "after.mp4")
    r = run(b, a, tmp_path / "run", Config(focal_px=600, sample_seconds=0.5))
    assert r["registration"]["valid"] and r["registration"]["inliers"] >= 6
    assert r["events"] == []
    assert r["geometry"]["before"]["registered_frames"] >= 2
    assert (tmp_path / "run/world.npz").exists()
    with pytest.raises(FileExistsError):
        run(b, a, tmp_path / "run")


def test_api_health_and_validation():
    from fastapi.testclient import TestClient
    from spatialmemory.app import app

    client = TestClient(app)
    assert client.get("/health").json()["status"] == "ok"
    assert "SpatialMemory-X" in client.get("/").text
    assert client.get("/jobs/not-a-job").status_code == 404
    assert client.post("/jobs").status_code == 422


def test_vfr_timestamps_preserved(tmp_path):
    import subprocess, cv2, numpy as np

    lines = []
    for i, duration in enumerate([0.04, 0.12, 0.08, 0.2]):
        image = tmp_path / f"{i}.png"
        cv2.imwrite(str(image), np.full((64, 64, 3), i * 50, np.uint8))
        lines.extend([f"file '{image}'", f"duration {duration}"])
    lines.append(f"file '{tmp_path / '3.png'}'")
    concat = tmp_path / "frames.txt"
    concat.write_text("\n".join(lines))
    video = tmp_path / "vfr.mkv"
    subprocess.run(
        [
            "ffmpeg",
            "-v",
            "error",
            "-f",
            "concat",
            "-safe",
            "0",
            "-i",
            str(concat),
            "-fps_mode",
            "vfr",
            "-c:v",
            "ffv1",
            str(video),
        ],
        check=True,
    )
    frames, manifest = ingest(
        video, tmp_path / "decoded", Config(sample_seconds=0.01), "before"
    )
    times = manifest["original_timestamps_seconds"]
    assert len(times) == manifest["decoded_frames"]
    assert len(set(round(b - a, 3) for a, b in zip(times, times[1:]))) > 1
    assert [f.timestamp for f in frames] == times


def test_orientation_metadata(tmp_path):
    import subprocess, cv2

    source = fixture.make_video(tmp_path / "source.mp4", frames=4)
    rotated = tmp_path / "rotated.mov"
    subprocess.run(
        [
            "ffmpeg",
            "-v",
            "error",
            "-display_rotation",
            "90",
            "-i",
            str(source),
            "-c",
            "copy",
            str(rotated),
        ],
        check=True,
    )
    frames, manifest = ingest(rotated, tmp_path / "rotated_frames", Config(), "before")
    image = cv2.imread(frames[0].path)
    assert image.shape[:2] == (640, 480)
    assert manifest["metadata"]["streams"][0]["side_data_list"][0]["rotation"] == 90


def test_frame_budget_covers_end_of_long_recording(tmp_path):
    path = fixture.make_video(tmp_path / "long.mp4", frames=60)
    frames, manifest = ingest(
        path, tmp_path / "frames", Config(max_frames=5, sample_seconds=0.05), "before"
    )
    assert len(frames) == 5 and frames[0].index == 0 and frames[-1].index == 59
    assert manifest["selection_limit_reached"] and manifest["decoded_frames"] == 60


def test_report_publication_failure_never_restores_as_complete(tmp_path, monkeypatch):
    from fastapi.testclient import TestClient
    from spatialmemory import app as runtime

    before = fixture.make_video(tmp_path / "before.mp4")
    after = fixture.make_video(tmp_path / "after.mp4")
    job = tmp_path / ("c" * 32)
    output = job / "output"
    original_replace = Path.replace

    def interrupted_publication(path, target):
        if path.name == "report.json.tmp":
            raise OSError("Simulated publication failure")
        return original_replace(path, target)

    monkeypatch.setattr(Path, "replace", interrupted_publication)
    with pytest.raises(OSError, match="publication failure"):
        run(before, after, output, Config(focal_px=600, sample_seconds=0.5))
    assert not (output / "report.json").exists()
    assert (output / "report.json.tmp").exists()
    monkeypatch.setattr(runtime, "ROOT", tmp_path)
    monkeypatch.setattr(runtime, "JOBS", {})
    client = TestClient(runtime.app)
    assert client.get("/jobs/" + job.name).json()["status"] == "failed"
    assert client.get("/jobs/" + job.name + "/report").status_code == 409
