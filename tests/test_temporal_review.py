import json
import cv2
import numpy as np
from spatialmemory.temporal_review import track_recording


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
