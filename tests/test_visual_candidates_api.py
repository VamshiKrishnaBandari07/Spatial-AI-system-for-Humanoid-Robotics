"""Dashboard visual-candidate and dense-world helpers."""

from pathlib import Path
import json
import numpy as np
from fastapi.testclient import TestClient


def test_world_prefers_dense_cloud(tmp_path, monkeypatch):
    from spatialmemory import app as app_module

    jid = "a" * 32
    root = tmp_path / jid / "output"
    root.mkdir(parents=True)
    np.savez_compressed(
        root / "world.npz",
        before=np.zeros((2, 3), np.float32),
        after=np.zeros((2, 3), np.float32),
        before_colors=np.zeros((2, 3), np.float32),
        after_colors=np.zeros((2, 3), np.float32),
    )
    np.savez_compressed(
        root / "world_dense.npz",
        before=np.ones((5, 3), np.float32),
        after=np.ones((7, 3), np.float32),
        before_colors=np.ones((5, 3), np.float32),
        after_colors=np.ones((7, 3), np.float32),
    )
    monkeypatch.setattr(app_module, "ROOT", tmp_path)
    client = TestClient(app_module.app)
    data = client.get(f"/jobs/{jid}/world").json()
    assert len(data["before"]) == 5
    assert len(data["after"]) == 7


def test_visual_candidates_lists_regions(tmp_path, monkeypatch):
    from spatialmemory import app as app_module

    jid = "b" * 32
    classical = tmp_path / jid / "classical"
    pair = classical / "b000_a000"
    pair.mkdir(parents=True)
    (pair / "comparison.jpg").write_bytes(b"x")
    (pair / "candidate_000.jpg").write_bytes(b"y")
    (pair / "before.jpg").write_bytes(b"z")
    (pair / "after_aligned.jpg").write_bytes(b"w")
    report = {
        "summary": {"candidate_region_instances": 1},
        "physical_change_verified": False,
        "limitations": ["visual only"],
        "groups": [
            {
                "before_timestamp": 0.0,
                "pairs": [
                    {
                        "folder": "b000_a000",
                        "after_id": "after:0",
                        "regions": [
                            {
                                "bbox_xyxy": [1, 2, 3, 4],
                                "area_pixels": 12,
                                "classification": "VISUAL_CHANGE_CANDIDATE",
                            }
                        ],
                    }
                ],
            }
        ],
    }
    (classical / "report.json").write_text(json.dumps(report))
    monkeypatch.setattr(app_module, "ROOT", tmp_path)
    client = TestClient(app_module.app)
    data = client.get(f"/jobs/{jid}/visual-candidates").json()
    assert data["total_candidates"] == 1
    assert data["candidates"][0]["crop"].endswith("candidate_000.jpg")
    assert data["physical_change_verified"] is False
