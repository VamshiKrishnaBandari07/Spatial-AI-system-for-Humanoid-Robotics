"""group_with_gallery must keep unconvertible evidence and explain why."""

from spatialmemory.contracts import Config
from spatialmemory.identity_integration import group_with_gallery
from spatialmemory.memory import Observation

BOX = [[-0.05, -0.05, 0.95], [0.05, 0.05, 1.05]]


def make(identifier, descriptor, x=0.0):
    return Observation(
        id=identifier, frame_id="f0", timestamp=0.0, label="cup",
        position=[x, 0.0, 1.0], descriptor=descriptor, viewpoint="v",
        provenance={"extent": BOX},
    )


def gallery_info(objects):
    return {row.id: row.provenance["identity_gallery"] for o in objects for row in o.observations}


def test_unusable_descriptor_is_retained_unresolved_with_reason():
    objects = group_with_gallery([make("good", [1.0, 0.0]), make("bad", None, x=1.0)], "before", Config())
    info = gallery_info(objects)
    assert set(info) == {"good", "bad"}, "no observation may be dropped"
    assert "conversion_error" not in info["good"]
    assert "descriptor" in info["bad"]["conversion_error"].lower()
    assert info["bad"]["status"] == "deferred" and info["bad"]["calibrated"] is False


def test_zero_descriptor_fails_closed_but_is_not_lost():
    objects = group_with_gallery([make("zero", [0.0, 0.0])], "after", Config())
    info = gallery_info(objects)
    assert list(info) == ["zero"] and "conversion_error" in info["zero"]
    assert not any(row.id == "zero" and o.id and not o.id.startswith("after_unresolved_")
                   for o in objects for row in o.observations)


def test_valid_observations_still_get_provisional_track():
    objects = group_with_gallery([make("ok", [1.0, 0.0])], "before", Config())
    info = gallery_info(objects)
    assert info["ok"]["track_id"] and "conversion_error" not in info["ok"]
