import pytest

from spatialmemory.association import semantic_label, pair_measurements
from types import SimpleNamespace


@pytest.mark.parametrize(
    "raw,expected",
    [
        ("bottle bottle", "bottle"),
        ("cup mug", "cup"),
        ("mug", "cup"),
        ("cup cup mug", "cup"),
        ("cup mug kettle", "cup mug kettle"),
        ("cup mug bowl", "cup mug bowl"),
        ("toiletry", "toiletry"),
        ("bottle", "bottle"),
    ],
)
def test_conservative_normalization(raw, expected):
    assert semantic_label(raw) == expected


def test_alias_compatibility_does_not_manufacture_identity():
    def obj(label):
        return SimpleNamespace(label=label, position=None, descriptor=[1, 0])

    assert pair_measurements(obj("cup mug"), obj("mug"))["semantic_compatible"]
    assert not pair_measurements(obj("cup mug kettle"), obj("cup"))[
        "semantic_compatible"
    ]
