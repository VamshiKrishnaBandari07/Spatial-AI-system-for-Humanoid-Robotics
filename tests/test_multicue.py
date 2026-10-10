"""Synthetic association/error-budget fixtures, never real-world accuracy labels."""

from types import SimpleNamespace
import numpy as np
import pytest
from spatialmemory.appearance import masked_crop
from spatialmemory.association import match_objects, unit_descriptor
from spatialmemory.contracts import Config
from spatialmemory.movement import bounded_movement


def obj(identifier="a", x=0, label="chair", descriptor=None, views=2, size=0.2):
    return SimpleNamespace(
        id=identifier,
        label=label,
        descriptor=descriptor if descriptor is not None else [1.0, 0.0, 0.0],
        position=[x, 0, 0],
        extent=[[x, 0, 0], [x + size, size, size]],
        distinct_views=views,
        position_spread=0.001,
    )


def cfg(**extra):
    return Config(association_mode="multiview", **extra)


def test_l2_normalization_invariant_to_positive_descriptor_scale():
    assert np.allclose(unit_descriptor([3, 4]), [0.6, 0.8])
    assert np.allclose(unit_descriptor([30, 40]), unit_descriptor([3, 4]))


@pytest.mark.parametrize("bad", [[0, 0], [np.nan, 1], [[1, 2]], []])
def test_invalid_descriptors_are_rejected(bad):
    with pytest.raises(ValueError):
        unit_descriptor(bad)


def test_mask_crop_preserves_pixel_axes_and_excludes_background():
    rgb = np.arange(6 * 8 * 3, dtype=np.uint8).reshape(6, 8, 3)
    mask = np.zeros((6, 8), bool)
    mask[2, 4] = True
    crop = masked_crop(rgb, mask, [3, 1, 7, 5])
    assert crop.shape == (4, 4, 3) and np.array_equal(crop[1, 1], rgb[2, 4])
    assert np.count_nonzero(crop) == 3
    with pytest.raises(ValueError):
        masked_crop(rgb, np.zeros((8, 6)), [3, 1, 7, 5])
    with pytest.raises(ValueError):
        masked_crop(rgb, mask, [-1, 0, 3, 4])


def test_global_matching_is_one_to_one_with_explicit_unmatched_objects():
    matches = match_objects(
        [obj("a", 0), obj("b", 2)],
        [obj("c", 0.05), obj("d", 2.05), obj("unmatched", 10)],
        cfg(association_search_radius=0.4),
    )
    assert {(m["before"], m["after"]) for m in matches} == {(0, 0), (1, 1)}
    assert all(m["identity_verified"] for m in matches)


def test_geometry_can_resolve_same_appearance_spatially_separated_objects():
    before = [obj("a", 0), obj("b", 2)]
    after = [obj("c", 0.2), obj("d", 2.2)]
    legacy = match_objects(before, after, Config())
    assert all(not m["identity_verified"] for m in legacy)
    matches = match_objects(before, after, cfg(association_search_radius=0.5))
    assert len(matches) == 2 and all(m["identity_verified"] for m in matches)


def test_identical_nearby_candidates_remain_ambiguous():
    matches = match_objects([obj("a", 0)], [obj("b", 0.01), obj("c", -0.01)], cfg())
    assert len(matches) == 1 and not matches[0]["identity_verified"]


def test_low_appearance_cannot_be_rescued_by_geometry_or_labels():
    assert match_objects([obj()], [obj(descriptor=[0, 1, 0])], cfg()) == []
    assert match_objects([obj()], [obj(label="table")], cfg()) == []


def test_multiview_gate_and_degenerate_shape_protect_against_false_identity():
    assert match_objects([obj(views=1)], [obj()], cfg()) == []
    assert match_objects([obj(size=0)], [obj()], cfg()) == []
    assert match_objects([obj()], [obj(descriptor=[1, 0])], cfg()) == []


def test_declared_error_budgets_support_large_synthetic_displacement_only():
    config = Config(
        movement_mode="bounded",
        voxel_size=0.01,
        movement_depth_error_bound=0.002,
        movement_registration_error_bound=0.005,
        movement_error_source="synthetic exact error budget, not real calibration",
    )
    assert bounded_movement(obj(), obj(x=0.08), config)[0] == "MOVED"
    assert bounded_movement(obj(), obj(x=0.02), config)[0] == "UNKNOWN"
    assert bounded_movement(obj(), obj(), config)[0] == "UNCHANGED"
    assert bounded_movement(obj(), obj(views=1), config)[0] == "UNKNOWN"
    decision, evidence = bounded_movement(
        obj(), obj(x=0.08), Config(movement_mode="bounded", voxel_size=0.01)
    )
    assert decision == "UNKNOWN" and not evidence["calibrated_covariance"]


@pytest.mark.parametrize("free_views", [0, 2])
def test_unresolved_identity_cannot_become_contradictory_added_removed(
    monkeypatch, free_views
):
    from spatialmemory.semantics import consensus
    from spatialmemory.contracts import Registration, Change
    import spatialmemory.semantics as semantics

    monkeypatch.setattr(
        semantics,
        "visibility_evidence",
        lambda *a, **k: {"supported_free_views": free_views, "views": []},
    )
    empty = SimpleNamespace(surfaces={}, poses={})
    a, b = obj(), obj("b", 0.05, descriptor=[0, 1, 0])
    for item in [a, b]:
        item.observations = []
    events, matches = consensus(
        [a],
        [b],
        empty,
        empty,
        Registration(
            valid=True,
            scale=1,
            rotation=np.eye(3).tolist(),
            translation=[0, 0, 0],
            inliers=30,
            correspondences=30,
            rmse=0.01,
            threshold=0.08,
            reason="synthetic fixture",
        ),
        cfg(),
    )
    assert not matches and len(events) == 2
    assert all(e.change == Change.UNKNOWN for e in events)
    assert all(e.evidence["unresolved_identity_candidates"] for e in events)


def test_matched_identity_is_consumed_once_and_missing_visibility_blocks_removal(
    monkeypatch,
):
    from spatialmemory.semantics import consensus
    from spatialmemory.contracts import Registration, Change
    import spatialmemory.semantics as semantics

    monkeypatch.setattr(
        semantics,
        "visibility_evidence",
        lambda *a, **k: {"supported_free_views": 0, "views": []},
    )
    empty = SimpleNamespace(surfaces={}, poses={})
    a, b = obj(), obj("b", 0.2)
    a.observations = []
    b.observations = []
    registration = Registration(
        valid=True,
        scale=1,
        rotation=np.eye(3).tolist(),
        translation=[0, 0, 0],
        inliers=30,
        correspondences=30,
        rmse=0.01,
        threshold=0.08,
        reason="synthetic fixture",
    )
    events, matches = consensus(
        [a], [b], empty, empty, registration, cfg(movement_mode="bounded")
    )
    assert len(events) == len(matches) == 1 and events[0].change == Change.UNKNOWN
    events, _ = consensus([a], [], empty, empty, registration, cfg())
    assert events[0].change == Change.UNKNOWN


def test_viewer_contains_only_computed_points_and_escapes_script_content(tmp_path):
    from spatialmemory.viewer import write_viewer

    report = {
        "scene_id": "</script><script>window.injected=true</script>",
        "events": [
            {
                "id": "x",
                "semantic_label": "chair",
                "change": "UNKNOWN",
                "before_position": [0, 0, 0],
                "after_position": None,
                "explanation": "Unknown",
                "evidence": {
                    "observations": [
                        {
                            "provenance": {
                                "evidence_file": "../../secret.png",
                                "mask_file": "safe_mask.png",
                            }
                        }
                    ]
                },
            }
        ],
    }
    path = write_viewer(tmp_path, report, np.ones((3, 3)), np.zeros((3, 3)))
    text = path.read_text()
    assert (
        report["scene_id"] not in text
        and "safe_mask.png" in text
        and "../../secret.png" not in text
    )
    assert "UNKNOWN" in text


def test_nonfinite_error_budgets_and_centroids_cannot_support_movement():
    from pydantic import ValidationError

    for value in [float("inf"), float("nan"), -1]:
        with pytest.raises(ValidationError):
            Config(movement_depth_error_bound=value)
    with pytest.raises(ValidationError):
        Config(movement_error_source="")
    bad = obj()
    bad.position = [float("nan"), 0, 0]
    configuration = Config(
        movement_depth_error_bound=0.002,
        movement_registration_error_bound=0.005,
        movement_error_source="synthetic",
    )
    decision, evidence = bounded_movement(bad, obj(), configuration)
    assert (
        decision == "UNKNOWN"
        and evidence["blocking_gate"] == "invalid_centroid_geometry"
    )
