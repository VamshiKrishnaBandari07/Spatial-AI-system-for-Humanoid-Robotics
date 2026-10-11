import json
import cv2
import numpy as np
from spatialmemory.temporal_review import track_recording
from spatialmemory.temporal_review import link_redetections
from spatialmemory.segmentation import propagate_mask


def test_shared_flow_cache_matches_uncached_and_invalidates_new_frames():
    image = np.random.default_rng(7).integers(0, 255, (64, 64, 3), dtype=np.uint8)
    current = np.roll(image, 2, axis=1)
    cache = {}
    for offset in (12, 30):
        mask = np.zeros((64, 64), bool)
        mask[offset : offset + 15, 15:30] = True
        expected, audit = propagate_mask(image, current, mask)
        actual, cached_audit = propagate_mask(image, current, mask, flow_cache=cache)
        np.testing.assert_array_equal(actual, expected)
        assert cached_audit == audit
    next_frame = np.roll(image, 5, axis=1)
    expected, _ = propagate_mask(current, next_frame, mask)
    actual, _ = propagate_mask(current, next_frame, mask, flow_cache=cache)
    np.testing.assert_array_equal(actual, expected)


def test_all_frames_preserve_timestamps_and_anchor_provenance(tmp_path):
    video = tmp_path / "clip.avi"
    writer = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*"MJPG"), 10, (64, 64))
    image = np.random.default_rng(7).integers(0, 255, (64, 64, 3), dtype=np.uint8)
    for _ in range(8):
        writer.write(image)
    writer.release()
    (tmp_path / "before").mkdir()
    times = [i / 10 for i in range(8)]
    (tmp_path / "before/manifest.json").write_text(
        json.dumps({"original_timestamps_seconds": times})
    )
    (tmp_path / "before_model_outputs.json").write_text(
        json.dumps([{"frame_id": "before:0", "detections": {"labels": ["cup"]}}])
    )
    evidence = tmp_path / "evidence/supporting_frames"
    evidence.mkdir(parents=True)
    mask = np.zeros((64, 64), np.uint8)
    mask[20:40, 20:40] = 255
    cv2.imwrite(str(evidence / "before_0_0_mask.png"), mask)
    result = track_recording(video, tmp_path, "before", width=64)
    rows = [json.loads(line) for line in open(result["diagnostics"])]
    assert [row["timestamp_seconds"] for row in rows] == times
    assert result["decoded_frames"] == 8
    assert rows[0]["tracks"][0]["status"] == "neural_anchor"
    assert rows[-1]["tracks"][0]["status"] == "flow_supported_2d"
    assert not any(row["physical_change_verified"] for row in rows)


def test_real_callback_links_supported_tracks(tmp_path):
    test_all_frames_preserve_timestamps_and_anchor_provenance(tmp_path)
    calls = []

    def redetect(image):
        calls.append(True)
        mask = np.zeros(image.shape[:2], bool)
        mask[20:40, 20:40] = True
        return [{"mask": mask, "descriptor": [1, 0], "label": "cup"}]

    result = track_recording(
        tmp_path / "clip.avi",
        tmp_path,
        "before",
        width=64,
        redetect=redetect,
        redetect_seconds=0.2,
    )
    assert len(calls) >= 3
    assert result["supported_temporal_identity_links"] >= 2


def test_ambiguous_and_wrong_category_links_rejected():
    mask = np.ones((5, 5), bool)
    active = {"a": mask, "b": mask}
    features = {"a": np.array([1.0, 0]), "b": np.array([1.0, 0])}
    assert (
        link_redetections(
            [{"mask": mask, "descriptor": [1, 0], "label": "cup"}],
            active,
            features,
            {"a": "cup", "b": "cup"},
        )
        == {}
    )

    assert (
        link_redetections(
            [{"mask": mask, "descriptor": [1, 0], "label": "bottle"}],
            {"a": mask},
            features,
            {"a": "cup"},
        )
        == {}
    )


def test_drift_triggers_redetection_without_reusing_identity(tmp_path, monkeypatch):
    test_all_frames_preserve_timestamps_and_anchor_provenance(tmp_path)
    monkeypatch.setattr(
        "spatialmemory.temporal_review.propagate_mask",
        lambda a, b, mask, **kwargs: (
            np.zeros_like(mask),
            {"requires_reinitialization": True},
        ),
    )

    def detect(image):
        return [
            {
                "mask": np.ones(image.shape[:2], bool),
                "descriptor": [1, 0],
                "label": "cup",
            }
        ]

    result = track_recording(
        tmp_path / "clip.avi", tmp_path, "before", width=64, redetect=detect
    )
    assert result["neural_redetection_calls"] == 2
    assert result["supported_temporal_identity_links"] == 0


def test_upload_profile_preserves_tracking_option():
    from spatialmemory.app import upload_config

    assert upload_config("vggt", True, True).temporal_tracking_enabled
    assert not upload_config("vggt", True).temporal_tracking_enabled
