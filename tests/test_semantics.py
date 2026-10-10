from types import SimpleNamespace
import numpy as np, pytest
from spatialmemory.contracts import Config, Registration, Change
from spatialmemory.memory import Observation
from spatialmemory.semantics import (
    independent_views,
    group_objects,
    consensus,
    common_frame,
    persist_memory,
    visibility_evidence,
    ObjectState,
)
from spatialmemory.segmentation import propagate_mask
from spatialmemory.registration import register


def observation(name, frame, position, camera, descriptor=None, label="chair"):
    return Observation(
        id=name,
        frame_id=frame,
        timestamp=0,
        label=label,
        position=position,
        descriptor=descriptor or [1, 0],
        viewpoint=frame,
        provenance={"camera_center": camera, "extent": None},
    )


def reg():
    return Registration(
        valid=True,
        scale=1,
        rotation=np.eye(3).tolist(),
        translation=[0, 0, 0],
        inliers=30,
        correspondences=30,
        rmse=0,
        threshold=0.08,
        reason="synthetic fixture",
    )


def object_state(name, position, views=2, descriptor=None):
    obs = [
        observation(name + "a", "0", position, [-1, 0, -1], descriptor),
        observation(name + "b", "1", position, [1, 0, -1], descriptor),
    ]
    return ObjectState(
        name, "chair", position, descriptor or [1, 0], obs, None, views, 0
    )


def test_hundred_duplicate_views_count_once():
    obs = [observation(str(i), str(i), [0, 0, 1], [0, 0, 0]) for i in range(100)]
    assert independent_views(obs, [0, 0, 1], 10) == 1
    obs.append(observation("new", "new", [0, 0, 1], [1, 0, 0]))
    assert independent_views(obs, [0, 0, 1], 10) == 2


def test_grouping_preserves_same_frame_identical_objects():
    obs = [
        observation("a", "f", [0, 0, 1], [0, 0, 0]),
        observation("b", "f", [0.1, 0, 1], [0, 0, 0]),
    ]
    assert len(group_objects(obs, "before", Config())) == 2


def test_matched_movement_and_unchanged():
    empty = SimpleNamespace(surfaces={}, poses={})
    for target, expected in [([0, 0, 1], Change.UNCHANGED), ([1, 0, 1], Change.MOVED)]:
        events, _ = consensus(
            [object_state("a", [0, 0, 1])],
            [object_state("b", target)],
            empty,
            empty,
            reg(),
            Config(),
        )
        assert (
            events[0].change == expected
            and events[0].evidence["posterior_probability"] is None
        )
        assert events[0].displacement_units != "metres"
        assert len(events[0].supporting_frames) == 2


def test_single_view_is_unknown_and_occlusion_not_removed():
    empty = SimpleNamespace(surfaces={}, poses={})
    events, _ = consensus(
        [object_state("a", [0, 0, 1], 1)],
        [object_state("b", [1, 0, 1], 1)],
        empty,
        empty,
        reg(),
        Config(),
    )
    assert events[0].change == Change.UNKNOWN
    events, _ = consensus(
        [object_state("a", [0, 0, 1])], [], empty, empty, reg(), Config()
    )
    assert events[0].change == Change.UNKNOWN


def test_identical_object_association_remains_ambiguous():
    empty = SimpleNamespace(surfaces={}, poses={})
    events, _ = consensus(
        [object_state("a", [0, 0, 1]), object_state("b", [1, 0, 1])],
        [object_state("c", [0, 0, 1]), object_state("d", [1, 0, 1])],
        empty,
        empty,
        reg(),
        Config(),
    )
    assert all(e.change == Change.UNKNOWN and e.contradictions for e in events)


def test_positive_predicted_free_space_multiview():
    poses = {}
    surfaces = {}
    K = np.array([[20.0, 0, 20], [0, 20, 20], [0, 0, 1]])
    for i, x in enumerate([-1, 1]):
        pose = np.eye(4)
        pose[0, 3] = -x
        poses[str(i)] = pose
        surfaces[str(i)] = {
            "depth": np.full((40, 40), 3.0),
            "confidence": np.ones((40, 40)),
            "intrinsics": K,
        }
    r = SimpleNamespace(poses=poses, surfaces=surfaces)
    obj = object_state("a", [0, 0, 2])
    v = visibility_evidence(obj, r, Config())
    assert v["supported_free_views"] == 2 and not v["metric"]
    events, _ = consensus([obj], [], r, r, reg(), Config())
    assert (
        events[0].change == Change.REMOVED and events[0].decision_status == "probable"
    )
    for s in surfaces.values():
        s["depth"][:] = 1
    events, _ = consensus([obj], [], r, r, reg(), Config())
    assert events[0].change == Change.UNKNOWN


def test_common_frame_and_persistent_memory(tmp_path):
    r = reg()
    r.scale = 2
    r.translation = [1, 0, 0]
    o = observation("a", "after:0", [0, 0, 1], [0, 0, 0])
    o = common_frame([o], r)[0]
    assert o.position == [1, 0, 2] and o.provenance["camera_center"] == [1, 0, 0]
    b = [object_state("stable", [0, 0, 1])]
    a = [object_state("stable", [0, 0, 1])]
    # Unique observation IDs for two recordings are mandatory.
    for obs in a[0].observations:
        obs.id = "after_" + obs.id
    payload = persist_memory(tmp_path, b, a, [])
    assert payload["before"]["objects"][0]["id"] == payload["after"]["objects"][0]["id"]
    assert (tmp_path / "spatial_memory.sqlite").exists()
    with pytest.raises(FileExistsError):
        persist_memory(tmp_path, b, a, [])


def test_mask_propagation_preserves_supported_translation():
    import cv2

    rng = np.random.default_rng(4)
    a = rng.integers(0, 256, (120, 120, 3), dtype=np.uint8)
    b = cv2.warpAffine(a, np.array([[1, 0, 3], [0, 1, 0]], float), (120, 120))
    mask = np.zeros((120, 120), bool)
    mask[40:80, 40:80] = True
    moved, info = propagate_mask(a, b, mask)
    assert moved[60, 63] and info["valid_fraction"] > 0.8
    with pytest.raises(ValueError):
        propagate_mask(a, b, np.zeros((2, 2), bool))


def test_registration_holdout_and_foreground_exclusion(monkeypatch):
    import spatialmemory.registration as module

    rng = np.random.default_rng(3)
    points = rng.normal(size=(40, 3))
    coordinates = np.array([[i, 0] for i in range(40)], float)
    features = {"f": (coordinates, np.zeros((40, 128)), None)}
    tracks = {("f", i): i for i in range(40)}
    scene = SimpleNamespace(points=points, features=features, tracks=tracks)
    monkeypatch.setattr(module, "match", lambda a, b: [(i, i) for i in range(40)])
    mask = np.zeros((1, 40), bool)
    mask[:, :5] = True
    r = register(scene, scene, Config(), {"f": mask}, {"f": mask})
    assert r.valid and r.heldout_count > 0 and r.foreground_matches_excluded == 5
    mask[:] = True
    assert not register(scene, scene, Config(), {"f": mask}, {"f": mask}).valid


def test_likelihood_calibration_requires_labels_and_deduplicates_groups():
    from spatialmemory.calibration import LikelihoodCalibration
    from spatialmemory.evidence import Evidence, fuse

    rows = []
    for truth in [False, True]:
        for i in range(10):
            rows.append(
                {
                    "hypothesis_id": str(truth) + str(i),
                    "correlation_group": str(i),
                    "signal": "free-space",
                    "change_present": truth,
                    "signal_present": truth if i < 8 else not truth,
                }
            )
    fit = LikelihoodCalibration.fit(rows + rows, "synthetic labeled fixtures")
    assert fit.training_groups == 20 and fit.likelihood_ratio("free-space", True) > 0
    result = fuse([Evidence("view1", fit.likelihood_ratio("free-space", True), "fit")])
    assert (
        result["posterior_under_supplied_likelihoods"] > 0.5
        and not fit.audit()["calibrated_for_deployment"]
    )
    with pytest.raises(ValueError):
        LikelihoodCalibration.fit(rows[:10], "incomplete")


def test_output_lock_rejects_concurrent_writer(tmp_path):
    import fcntl
    from spatialmemory.pipeline import run

    with (tmp_path / ".run.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        with pytest.raises(RuntimeError, match="active pipeline"):
            run("missing", "missing", tmp_path)


def test_cached_geometry_refuses_different_inputs_before_model_import(tmp_path):
    import json
    from spatialmemory.neural_geometry import reusable_predictions

    config = Config(geometry_backend="vggt")
    report = {
        "config": config.model_dump(),
        "inputs": [{"sha256": "before"}, {"sha256": "after"}],
    }
    (tmp_path / "report.json").write_text(json.dumps(report))
    with pytest.raises(ValueError, match="different videos"):
        reusable_predictions(tmp_path, config, {"sha256": "other"}, {"sha256": "after"})
    report["config"]["max_width"] = 100
    (tmp_path / "report.json").write_text(json.dumps(report))
    with pytest.raises(ValueError, match="configuration differs"):
        reusable_predictions(
            tmp_path, config, {"sha256": "before"}, {"sha256": "after"}
        )


def test_evidence_png_routes_do_not_escape(tmp_path, monkeypatch):
    from fastapi.testclient import TestClient
    import spatialmemory.app as module

    monkeypatch.setattr(module, "ROOT", tmp_path)
    jid = "a" * 32
    path = tmp_path / jid / "output/evidence/supporting_frames"
    path.mkdir(parents=True)
    import cv2

    cv2.imwrite(str(path / "before_0_1.png"), np.zeros((10, 10, 3), np.uint8))
    client = TestClient(module.app)
    assert client.get(f"/jobs/{jid}/evidence/before_0_1.png").status_code == 200
    assert client.get(f"/jobs/{jid}/evidence/private.txt").status_code == 404


def test_detector_revision_must_be_immutable(tmp_path):
    from spatialmemory.models import Expert, ModelSpec

    with pytest.raises(ValueError):
        Expert(ModelSpec("facebook/dinov2-small", "v1", "appearance"), tmp_path)
