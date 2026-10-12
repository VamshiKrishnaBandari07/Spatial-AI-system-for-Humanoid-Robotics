from PIL import Image
from spatialmemory.multiscale_detection import multiscale_detect


class Detector:
    def __init__(self):
        self.calls = []

    def infer(self, image, prompt):
        self.calls.append(image.size)
        return {
            "boxes": [[10, 20, 40, 70]],
            "labels": ["bottle"],
            "raw_scores": [0.6],
            "provenance": {"model": "synthetic-test-double"},
            "calibrated": False,
        }


def test_tiles_preserve_native_coordinates_and_call_budget():
    detector = Detector()
    result = multiscale_detect(
        detector, Image.new("RGB", (464, 832)), "bottle.", 512, 2
    )
    assert len(detector.calls) == 3
    assert result["provenance"]["crop_rectangles"][0] == [0, 0, 464, 832]
    assert all(
        0 <= box[0] < box[2] <= 464 and 0 <= box[1] < box[3] <= 832
        for box in result["boxes"]
    )
    assert max(box[1] for box in result["boxes"]) > 20
    assert not result["calibrated"]


def test_small_input_does_not_repeat_inference():
    detector = Detector()
    result = multiscale_detect(detector, Image.new("RGB", (200, 200)), "bottle.")
    assert len(detector.calls) == 1 and len(result["boxes"]) == 1


def test_candidate_crop_refinement_is_bounded_and_mapped():
    detector = Detector()
    result = multiscale_detect(detector, Image.new("RGB", (464, 832)),
                              "bottle.", 512, 2, max_refinements=1)
    assert len(detector.calls) == 4
    assert len(result["provenance"]["refinement_rectangles"]) == 1
    assert not result["calibrated"]
