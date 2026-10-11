import pytest
from spatialmemory.detection_selection import select_detections


def test_duplicates_do_not_exhaust_budget():
    boxes = [[0, 0, 10, 10], [0, 0, 10, 10], [20, 20, 25, 25]]
    assert select_detections(
        boxes, [0.9, 0.8, 0.7], ["cup mug", "mug", "bottle"], 0.35, 2
    ) == [0, 2]


def test_neighboring_identical_objects_remain_separate():
    assert select_detections(
        [[0, 0, 10, 10], [8, 0, 18, 10]], [0.9, 0.8], ["cup", "cup"], 0.35, 2
    ) == [0, 1]


def test_conflicting_labels_are_not_silently_resolved():
    assert select_detections(
        [[0, 0, 10, 10]] * 2, [0.9, 0.8], ["cup", "kettle"], 0.35, 2
    ) == [0, 1]


@pytest.mark.parametrize(
    "box,score", [([0, 0, 0, 2], 0.8), ([0, 0, 2, 2], float("nan"))]
)
def test_invalid_proposals_fail(box, score):
    with pytest.raises(ValueError):
        select_detections([box], [score], ["cup"], 0.35, 2)
