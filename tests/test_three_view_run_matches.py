"""Feature ambiguity is rejected before evaluating geometry."""
import importlib.util
from pathlib import Path
import numpy as np

spec = importlib.util.spec_from_file_location('three_view_runner', Path(__file__).parents[1] / 'scripts/diagnose_three_view_run.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def test_mutual_ratio_rejects_identical_lookalikes():
    descriptors = np.zeros((3, 128), dtype=np.float32)
    assert module.mutual_matches(descriptors, descriptors) == {}


def test_mutual_ratio_preserves_unique_correspondences():
    descriptors = np.eye(4, 128, dtype=np.float32)
    assert module.mutual_matches(descriptors, descriptors) == {i: i for i in range(4)}
    assert module.mutual_matches(None, descriptors) == {}
