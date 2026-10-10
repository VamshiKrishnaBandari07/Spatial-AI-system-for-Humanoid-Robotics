import numpy as np, pytest
from spatialmemory.registration import transform, ransac
from spatialmemory.visibility import visibility
from spatialmemory.change import compare, classify_object
from spatialmemory.contracts import Config, Change
from spatialmemory.evidence import Evidence, fuse, rank_views
from spatialmemory.navigation import plan


def test_sim3_outliers():
    rng = np.random.default_rng(2)
    x = rng.normal(size=(100, 3))
    R = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]])
    y = transform(x, 2, R, np.array([1, 2, 3.0]))
    y[:20] = rng.normal(size=(20, 3)) * 20
    reg, mask = ransac(x, y, 0.001)
    assert (
        reg.valid
        and reg.inliers == 80
        and abs(reg.scale - 2) < 1e-10
        and reg.rmse < 1e-10
    )
    np.testing.assert_allclose(reg.rotation, R, atol=1e-10)


def test_degenerate_registration():
    x = np.arange(10)[:, None] * np.ones((1, 3))
    reg, _ = ransac(x, x)
    assert not reg.valid


def test_visibility_states():
    K = np.eye(3)
    pose = np.eye(4)
    depth = np.array([[2.0]])
    assert visibility(
        [[0, 0, 2], [0, 0, 1], [0, 0, 3], [10, 10, 1]], depth, K, pose
    ) == ["OCCUPIED", "FREE", "OCCLUDED", "UNKNOWN"]
    assert visibility([[0, 0, 2]], np.zeros((1, 1)), K, pose) == ["UNKNOWN"]


def test_sparse_absence_is_not_removal():
    rng = np.random.default_rng(4)
    b = rng.normal(0, 0.01, (20, 3))
    a = b + 10
    reg, _ = ransac(b, b, 0.01)
    events = compare(b, a, reg, Config(min_cluster=3))
    assert len(events) == 2 and all(e.change == Change.UNKNOWN for e in events)
    events = compare(
        b,
        a,
        reg,
        Config(min_cluster=3),
        free_before=lambda p: True,
        free_after=lambda p: True,
    )
    assert {e.change for e in events} == {Change.ADDED, Change.REMOVED}


def test_no_change():
    rng = np.random.default_rng(4)
    p = rng.normal(size=(30, 3))
    reg, _ = ransac(p, p, 0.01)
    assert compare(p, p, reg, Config()) == []


@pytest.mark.parametrize(
    "before,after,identity,free,expected",
    [
        (None, [0, 0, 0], False, True, Change.ADDED),
        ([0, 0, 0], None, False, True, Change.REMOVED),
        ([0, 0, 0], None, False, False, Change.UNKNOWN),
        ([0, 0, 0], [1, 0, 0], True, False, Change.MOVED),
        ([0, 0, 0], [0, 0, 0], True, False, Change.UNCHANGED),
        ([0, 0, 0], [1, 0, 0], False, False, Change.UNKNOWN),
    ],
)
def test_object_decisions(before, after, identity, free, expected):
    assert classify_object(before, after, identity, free, 0.2, 0.05) == expected


def test_correlated_evidence_and_contradiction():
    one = fuse([Evidence("same-view", 2, "detector")])
    many = fuse([Evidence("same-view", 2, "detector")] * 100)
    assert one == many
    assert fuse([Evidence("a", 2, "x"), Evidence("b", -2, "y")])["contradiction"]
    assert not one["calibrated"]


def test_adaptive_budget():
    c = [
        {"quality": 1, "viewpoint": "a"},
        {"quality": 0.9, "viewpoint": "a"},
        {"quality": 0.8, "viewpoint": "b"},
    ]
    assert len(rank_views(c, set(), 2)) == 2


def test_navigation_unknown_blocks():
    grid = np.zeros((5, 5), int)
    grid[2, :] = -1
    assert plan(grid, (0, 0), (4, 4)) is None
    grid[2, 2] = 0
    path = plan(grid, (0, 0), (4, 4))
    assert path and (2, 2) in path


def test_config_rejects_typo():
    with pytest.raises(ValueError):
        Config(mxa_frames=4)
