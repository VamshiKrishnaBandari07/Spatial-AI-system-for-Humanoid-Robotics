"""Synthetic camera tracking through a fully occluded frame, not benchmark accuracy."""

from pathlib import Path
import importlib.util
import cv2
import numpy as np
from spatialmemory.contracts import Config
from spatialmemory.ingestion import ingest
from spatialmemory.geometry import reconstruct


def test_failed_frame_does_not_poison_later_camera_tracking(tmp_path):
    spec = importlib.util.spec_from_file_location(
        "synthetic_fixture", Path(__file__).parents[1] / "scripts/synthetic_fixture.py"
    )
    fixture = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(fixture)
    video = fixture.make_video(tmp_path / "scan.mp4")
    config = Config(focal_px=600, sample_seconds=0.5)
    frames, _ = ingest(video, tmp_path / "frames", config, "scan")
    assert len(frames) >= 5
    failed = frames[2]
    image = cv2.imread(failed.path)
    assert cv2.imwrite(failed.path, np.zeros_like(image))
    result = reconstruct(frames, config)
    assert failed.id not in result.poses
    assert frames[3].id in result.poses
    assert frames[-1].id in result.poses
    assert any(
        row["frame_id"] == frames[3].id and row["anchor_frame_id"] == frames[1].id
        for row in result.diagnostics["recovery_anchor_attempts"]
    )
