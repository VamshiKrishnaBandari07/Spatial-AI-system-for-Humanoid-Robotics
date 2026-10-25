"""Unit tests for LoFTR diagnostic filters; no checkpoint or private frames required."""

import json
from pathlib import Path

import numpy as np
import pytest
from spatialmemory.matching_diagnostics import (
    PRODUCTION_MATCH_RETURN_CAP,
    PRODUCTION_SCORE_THRESHOLD,
    filter_stages,
    resolve_after_frame,
    score_histogram,
)


def test_score_histogram_buckets():
    hist = score_histogram([0.1, 0.25, 0.85, 0.9, float("nan")])
    assert hist["[0.0,0.2)"] == 1
    assert hist["[0.2,0.4)"] == 1
    assert hist["[0.8,1.0)"] == 2


def test_filter_stages_require_literal_threshold_and_masks():
    pa = np.array([[10.0, 10.0], [20.0, 20.0], [30.0, 30.0]])
    pb = np.array([[11.0, 11.0], [21.0, 21.0], [31.0, 31.0]])
    scores = np.array([0.2, 0.85, 0.95])
    mask = np.zeros((40, 40), np.uint8)
    mask[8:25, 8:25] = 255
    report = filter_stages(pa, pb, scores, mask, mask)
    assert report["score_threshold"] == PRODUCTION_SCORE_THRESHOLD
    assert report["score_threshold_is_calibrated_probability"] is False
    assert report["stages"]["model_returned"] == 3
    assert report["stages"]["score_at_or_above_threshold"] == 2
    # Third point is outside the mask even though score is high.
    assert report["stages"]["accepted"] == 1
    assert report["stages"]["accepted_returned"] == 1
    assert report["accepted_scores"] == [0.85]


def test_empty_model_return_stays_zero():
    mask = np.ones((8, 8), np.uint8)
    report = filter_stages(np.zeros((0, 2)), np.zeros((0, 2)), np.zeros(0), mask, mask)
    assert report["stages"]["accepted"] == 0
    assert report["stages"]["accepted_returned"] == 0
    assert report["accepted_native_a"] == []


def test_accepted_returned_respects_production_cap():
    """Uncapped survivors can exceed 100; returned list matches production cap."""
    n = PRODUCTION_MATCH_RETURN_CAP + 25
    coords = np.column_stack([np.full(n, 5.0), np.full(n, 5.0)])
    scores = np.linspace(0.81, 0.99, n)
    mask = np.ones((16, 16), np.uint8) * 255
    report = filter_stages(coords, coords, scores, mask, mask)
    assert report["stages"]["accepted"] == n
    assert report["stages"]["accepted_returned"] == PRODUCTION_MATCH_RETURN_CAP
    assert len(report["accepted_scores"]) == PRODUCTION_MATCH_RETURN_CAP
    # Production agreement must use capped count, not uncapped accepted.
    assert report["stages"]["accepted_returned"] != report["stages"]["accepted"]


def test_resolve_after_frame_uses_ingestion_filename(tmp_path: Path):
    after = tmp_path / "after"
    after.mkdir()
    target = after / "after_001020.jpg"
    target.write_bytes(b"fake")
    assert resolve_after_frame(tmp_path, 1020) == target.resolve()


def test_resolve_after_frame_manifest_relative_to_after_not_cwd(tmp_path: Path, monkeypatch):
    after = tmp_path / "after"
    after.mkdir()
    target = after / "after_002040.jpg"
    target.write_bytes(b"fake")
    (after / "manifest.json").write_text(
        json.dumps(
            {
                "frames": [
                    {
                        "id": "after:2040",
                        "index": 2040,
                        # Relative path as stored without absolute root.
                        "path": "after_002040.jpg",
                    }
                ]
            }
        )
    )
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    monkeypatch.chdir(elsewhere)
    found = resolve_after_frame(tmp_path, 2040)
    assert found == target.resolve()
    assert found.is_relative_to(after.resolve())


def test_resolve_after_frame_missing_raises(tmp_path: Path):
    (tmp_path / "after").mkdir()
    with pytest.raises(FileNotFoundError, match="AFTER frame 99"):
        resolve_after_frame(tmp_path, 99)
