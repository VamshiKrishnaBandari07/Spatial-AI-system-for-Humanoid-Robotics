"""Synthetic identity regression tests; not real-world accuracy measurements."""

from dataclasses import replace
import numpy as np
import pytest
from spatialmemory.identity import (
    IdentityObservation,
    IdentitySettings,
    IdentityGallery,
    joint_assignment,
    iou_3d,
)
from spatialmemory.contracts import Config
from spatialmemory.memory import Observation
from spatialmemory.semantics import group_objects
from spatialmemory.association import match_objects


def obj(name, x=0.0, label="cup", embedding=(1.0, 0.0), position=True):
    return IdentityObservation(
        name,
        label,
        np.asarray(embedding),
        np.array([x, 0.0, 1.0]) if position else None,
        np.array([[x - 0.05, -0.05, 0.95], [x + 0.05, 0.05, 1.05]]),
    )


def test_cost_normalization_iou_and_transform_units():
    a, b = obj("a"), obj("b", 0.05)
    assert iou_3d(a.extent, a.extent) == 1
    assert iou_3d(a.extent, obj("far", 3).extent) == 0
    result = joint_assignment([a], [b], IdentitySettings())
    assert result.links == [(0, 0)] and 0 <= result.costs[0, 0] <= 1
    scale = 3
    scaled_a = replace(a, position=a.position * scale, extent=a.extent * scale)
    scaled_b = replace(b, position=b.position * scale, extent=b.extent * scale)
    np.testing.assert_allclose(
        result.costs,
        joint_assignment(
            [scaled_a],
            [scaled_b],
            IdentitySettings(radius=scale, proximity=0.2 * scale),
        ).costs,
    )


def test_nearby_lookalikes_rejected_before_assignment_both_sides():
    settings = IdentitySettings()
    for a, b in [
        ([obj("a")], [obj("b", 0.01), obj("c", -0.01)]),
        ([obj("a", 0.01), obj("d", -0.01)], [obj("b")]),
    ]:
        result = joint_assignment(a, b, settings)
        assert not result.links
        assert result.deferred_columns
        assert any(r["gate"] == "lookalike_ambiguity" for r in result.diagnostics)


def test_separated_lookalikes_and_unmatched_do_not_force_assignment():
    result = joint_assignment(
        [obj("a"), obj("b", 2)],
        [obj("c", 0.01), obj("d", 2.01), obj("new", 10)],
        IdentitySettings(radius=0.5),
    )
    assert result.links == [(0, 0), (1, 1)]
    assert np.isinf(result.costs[:, 2]).all()


def test_dynamic_floor_and_soft_class_penalty():
    vector = (0.9, np.sqrt(1 - 0.9**2))
    settings = IdentitySettings(iou_weight=0)
    near = joint_assignment([obj("a")], [obj("b", embedding=vector)], settings)
    far = joint_assignment([obj("a")], [obj("b", 0.3, embedding=vector)], settings)
    assert not near.links and far.links
    same = joint_assignment([obj("a")], [obj("b")], settings)
    confused = joint_assignment([obj("a")], [obj("b", label="bowl")], settings)
    assert confused.links and confused.costs[0, 0] > same.costs[0, 0]
    assert not joint_assignment(
        [obj("a")], [obj("b", label="bowl", embedding=vector)], settings
    ).links


def test_missing_geometry_and_dimension_mismatch_fail_closed():
    for b in [obj("b", position=False), obj("b", embedding=(1.0, 0.0, 0.0))]:
        assert not joint_assignment([obj("a")], [b], IdentitySettings()).links


def test_gallery_occlusion_handoff_ema_and_confirmation():
    gallery = IdentityGallery(IdentitySettings(memory_batches=2), "world")

    def update(objects, t, s="before"):
        return gallery.update(objects, timestamp=t, session=s, coordinate_frame="world")

    identifier = update([obj("a")], 0)["a"]
    assert not gallery.tracks[identifier].confirmed
    update([], 1)
    descriptor = (0.99, np.sqrt(1 - 0.99**2))
    assert update([obj("b", embedding=descriptor)], 2)["b"] == identifier
    track = gallery.tracks[identifier]
    assert track.confirmed and np.isclose(np.linalg.norm(track.prototype.descriptor), 1)
    assert 0 < track.prototype.descriptor[1] < descriptor[1]
    assert update([obj("c")], 0, "after")["c"] == identifier


def test_expired_track_is_not_reacquired():
    gallery = IdentityGallery(IdentitySettings(memory_batches=0), "world")
    old = gallery.update(
        [obj("a")], timestamp=0, session="s", coordinate_frame="world"
    )["a"]
    gallery.update([], timestamp=1, session="s", coordinate_frame="world")
    new = gallery.update(
        [obj("b")], timestamp=2, session="s", coordinate_frame="world"
    )["b"]
    assert new != old


def test_ambiguous_handoff_does_not_update_or_spawn_identity():
    gallery = IdentityGallery(IdentitySettings(), "world")
    gallery.update(
        [obj("a", 0.01), obj("b", -0.01)],
        timestamp=0,
        session="s",
        coordinate_frame="world",
    )
    prototypes = [t.prototype.descriptor.copy() for t in gallery.tracks.values()]
    result = gallery.update(
        [obj("c")], timestamp=1, session="s", coordinate_frame="world"
    )
    assert result == {"c": None} and len(gallery.tracks) == 2
    for previous, track in zip(prototypes, gallery.tracks.values()):
        np.testing.assert_array_equal(previous, track.prototype.descriptor)
        assert track.hits == 1


def test_coordinate_frame_and_time_guard():
    gallery = IdentityGallery(IdentitySettings(), "world")
    gallery.update([], timestamp=0, session="s", coordinate_frame="world")
    with pytest.raises(ValueError, match="coordinate frame"):
        gallery.update([], timestamp=1, session="s", coordinate_frame="camera")
    with pytest.raises(ValueError, match="increase"):
        gallery.update([], timestamp=0, session="s", coordinate_frame="world")


@pytest.mark.parametrize(
    "kwargs",
    [
        {"radius": 0},
        {"ema_alpha": 2},
        {"geo_weight": -1},
        {"max_cost": np.nan},
        {"confirmation_hits": 0},
    ],
)
def test_settings_reject_invalid_values(kwargs):
    with pytest.raises(ValueError):
        IdentitySettings(**kwargs)


def test_gallery_evidence_feeds_central_3d_grouping_and_cross_recording_match():
    config = Config(association_mode="joint")

    def observations(state):
        result = []
        for i, camera in enumerate([[-1, 0, 0], [1, 0, 0]]):
            result.append(
                Observation(
                    id=f"{state}_{i}",
                    frame_id=f"{state}:{i}",
                    timestamp=i,
                    label="cup",
                    position=[0, 0, 1],
                    descriptor=[1, 0],
                    viewpoint=str(i),
                    provenance={
                        "camera_center": camera,
                        "extent": [[-0.05, -0.05, 0.95], [0.05, 0.05, 1.05]],
                    },
                )
            )
        return result

    a = group_objects(observations("before"), "before", config)
    b = group_objects(observations("after"), "after", config)
    assert len(a) == len(b) == 1 and a[0].distinct_views == 2
    assert all(
        o.provenance["identity_gallery"]["status"] == "confirmed_local"
        for o in a[0].observations
    )
    matches = match_objects(a, b, config)
    assert len(matches) == 1 and matches[0]["identity_verified"]
    assert matches[0]["method"] == "joint"
    assert not matches[0]["score_is_probability"]


def test_gallery_checkpoint_preserves_identity_for_next_recording(tmp_path):
    gallery = IdentityGallery(IdentitySettings(), "shared_world", "scene_01")
    identifier = gallery.update(
        [obj("before_0")],
        timestamp=0,
        session="before",
        coordinate_frame="shared_world",
    )["before_0"]
    gallery.update(
        [obj("before_1")],
        timestamp=1,
        session="before",
        coordinate_frame="shared_world",
    )
    path = tmp_path / "gallery.json"
    gallery.save(path)
    restored = IdentityGallery.load(path)
    assert restored.tracks[identifier].confirmed
    result = restored.update(
        [obj("after_0")], timestamp=0, session="after", coordinate_frame="shared_world"
    )
    assert result["after_0"] == identifier
    with pytest.raises(ValueError, match="Repeated observation"):
        restored.update(
            [obj("after_0")],
            timestamp=1,
            session="after",
            coordinate_frame="shared_world",
        )


def test_confirmation_gate_prevents_premature_global_link():
    from types import SimpleNamespace

    config = Config(association_mode="joint", min_object_views=1)
    a = SimpleNamespace(
        id="a",
        label="cup",
        descriptor=[1.0, 0.0],
        position=[0.0, 0.0, 1.0],
        distinct_views=1,
        identity_confirmed=False,
        extent=[[-0.05, -0.05, 0.95], [0.05, 0.05, 1.05]],
    )
    audit = []
    assert not match_objects([a], [a], config, diagnostics=audit)
    assert audit[0]["gate"] == "unconfirmed_temporal_identity"


def test_object_state_identity_confirmed_defaults_false():
    """Missing or default confirmation must not pass joint cross-recording gates."""
    from types import SimpleNamespace
    from spatialmemory.semantics import ObjectState

    assert (
        ObjectState("x", "cup", [0, 0, 1], [1.0, 0.0], [], None).identity_confirmed
        is False
    )
    config = Config(association_mode="joint", min_object_views=1)
    # Attribute absent entirely (legacy / incomplete objects).
    bare = SimpleNamespace(
        id="bare",
        label="cup",
        descriptor=[1.0, 0.0],
        position=[0.0, 0.0, 1.0],
        distinct_views=2,
        extent=[[-0.05, -0.05, 0.95], [0.05, 0.05, 1.05]],
    )
    audit = []
    assert not match_objects([bare], [bare], config, diagnostics=audit)
    assert audit[0]["gate"] == "unconfirmed_temporal_identity"
    # Explicit True still links when views suffice.
    confirmed = SimpleNamespace(**{**bare.__dict__, "identity_confirmed": True})
    assert match_objects([confirmed], [confirmed], config)


def test_joint_consensus_persists_same_identity_across_recordings(tmp_path):
    from types import SimpleNamespace
    from spatialmemory.contracts import Registration, Change
    from spatialmemory.semantics import consensus, persist_memory
    from spatialmemory.memory import Memory

    config = Config(association_mode="joint")
    groups = {}
    for state in ["before", "after"]:
        observations = [
            Observation(
                id=f"{state}_{i}",
                frame_id=f"{state}:{i}",
                timestamp=i,
                label="cup",
                position=[0, 0, 1],
                descriptor=[1, 0],
                viewpoint=str(i),
                provenance={
                    "camera_center": [x, 0, 0],
                    "extent": [[-0.05, -0.05, 0.95], [0.05, 0.05, 1.05]],
                },
            )
            for i, x in enumerate([-1, 1])
        ]
        groups[state] = group_objects(
            observations, state, config, tmp_path / f"{state}_checkpoint.json"
        )
    registration = Registration(
        valid=True,
        scale=1,
        rotation=np.eye(3).tolist(),
        translation=[0, 0, 0],
        inliers=20,
        correspondences=20,
        rmse=0,
        threshold=0.08,
        reason="Synthetic exact alignment",
    )
    reconstruction = SimpleNamespace(surfaces={}, poses={})
    events, matches = consensus(
        groups["before"],
        groups["after"],
        reconstruction,
        reconstruction,
        registration,
        config,
    )
    assert len(matches) == 1 and events[0].change == Change.UNCHANGED
    assert groups["before"][0].id == groups["after"][0].id
    persist_memory(tmp_path, groups["before"], groups["after"], events)
    memory = Memory(tmp_path / "spatial_memory.sqlite")
    try:
        assert len({o.object_id for o in memory.observations()}) == 1
    finally:
        memory.close()
