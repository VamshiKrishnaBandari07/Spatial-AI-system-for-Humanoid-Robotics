"""Unit tests for LoFTR diagnostic filters; no checkpoint or private frames required."""

import numpy as np
from spatialmemory.matching_diagnostics import (
    PRODUCTION_SCORE_THRESHOLD,
    filter_stages,
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
    assert report["accepted_scores"] == [0.85]


def test_empty_model_return_stays_zero():
    mask = np.ones((8, 8), np.uint8)
    report = filter_stages(np.zeros((0, 2)), np.zeros((0, 2)), np.zeros(0), mask, mask)
    assert report["stages"]["accepted"] == 0
    assert report["accepted_native_a"] == []
