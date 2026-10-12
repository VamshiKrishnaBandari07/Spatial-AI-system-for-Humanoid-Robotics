import json
import cv2
import numpy as np
from spatialmemory.object_review import create_object_review


def test_review_preserves_ambiguous_labels_and_crops(tmp_path):
    directory = tmp_path / "evidence/supporting_frames"
    directory.mkdir(parents=True)
    cv2.imwrite(str(directory / "before_0_0.png"), np.zeros((20, 20, 3), np.uint8))
    (tmp_path / "before_model_outputs.json").write_text(
        json.dumps(
            [
                {
                    "frame_id": "before:0",
                    "detections": {
                        "boxes": [[2, 3, 12, 13]],
                        "labels": ["cup mug kettle"],
                        "raw_scores": [0.8],
                    },
                }
            ]
        )
    )
    page = create_object_review(tmp_path)
    rows = json.loads((page.parent / "observations.json").read_text())
    assert rows[0]["display_name"] == "Unidentified object"
    assert rows[0]["identity_verified"] is False
    assert cv2.imread(str(page.parent / rows[0]["crop"])).shape == (10, 10, 3)


def test_unprocessed_proposal_gets_crop_without_verified_name(tmp_path):
    (tmp_path / "before").mkdir()
    cv2.imwrite(
        str(tmp_path / "before/before_000000.jpg"), np.zeros((20, 20, 3), np.uint8)
    )
    (tmp_path / "before_model_outputs.json").write_text(
        json.dumps(
            [
                {
                    "frame_id": "before:0",
                    "perception_size": [20, 20],
                    "detections": {
                        "boxes": [[2, 3, 12, 13]],
                        "labels": ["bottle"],
                        "raw_scores": [0.32],
                    },
                }
            ]
        )
    )
    page = create_object_review(tmp_path)
    row = json.loads((page.parent / "observations.json").read_text())[0]
    assert row["display_name"] == "Unidentified object"
    assert row["processing_status"] == "unverified_proposal"
    assert (page.parent / row["crop"]).is_file()


def test_review_uses_final_semantics_instead_of_rejected_detector_label(tmp_path):
    import sqlite3

    directory = tmp_path / 'evidence/supporting_frames'
    directory.mkdir(parents=True)
    cv2.imwrite(str(directory / 'before_0_0.png'), np.zeros((20, 20, 3), np.uint8))
    (tmp_path / 'before_model_outputs.json').write_text(json.dumps([{
        'frame_id': 'before:0',
        'detections': {'boxes': [[0, 0, 15, 15]], 'labels': ['cup'], 'raw_scores': [0.8]},
    }]))
    with sqlite3.connect(tmp_path / 'spatial_memory.sqlite') as connection:
        connection.execute('CREATE TABLE observations (data TEXT)')
        connection.execute('INSERT INTO observations VALUES (?)', (json.dumps({
            'id': 'before_0_0', 'label': 'unidentified object',
        }),))
    page = create_object_review(tmp_path)
    row = json.loads((page.parent / 'observations.json').read_text())[0]
    assert row['display_name'] == 'Unidentified Object'
    assert row['raw_label'] == 'cup'
    assert row['label_source'] == 'stored_observation'
    with sqlite3.connect(tmp_path / 'spatial_memory.sqlite') as connection:
        connection.execute('UPDATE observations SET data=?', (json.dumps({
            'id': 'before_0_0', 'label': 'toilet paper roll',
        }),))
    page = create_object_review(tmp_path)
    row = json.loads((page.parent / 'observations.json').read_text())[0]
    assert row['display_name'] == 'Toilet Paper Roll'
