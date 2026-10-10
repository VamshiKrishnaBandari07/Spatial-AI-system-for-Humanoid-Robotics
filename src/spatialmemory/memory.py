"""Versioned scene memory keeps raw observations distinct from beliefs."""

import json, sqlite3
from pydantic import BaseModel, Field
from scipy.optimize import linear_sum_assignment
import numpy as np


class Observation(BaseModel):
    id: str
    object_id: str | None = None
    frame_id: str
    timestamp: float
    label: str
    position: list[float] | None = None
    covariance: list[list[float]] | None = None
    descriptor: list[float] | None = None
    viewpoint: str
    provenance: dict = Field(default_factory=dict)


class Memory:
    def __init__(self, path):
        self.db = sqlite3.connect(path)
        self.db.execute(
            "CREATE TABLE IF NOT EXISTS observations (id TEXT PRIMARY KEY, data TEXT NOT NULL)"
        )
        self.db.execute(
            "CREATE TABLE IF NOT EXISTS worlds (version TEXT PRIMARY KEY, data TEXT NOT NULL)"
        )
        self.db.commit()

    def observe(self, observation: Observation):
        with self.db:
            self.db.execute(
                "INSERT INTO observations VALUES (?,?)",
                (observation.id, observation.model_dump_json()),
            )

    def snapshot(self, version, beliefs):
        with self.db:
            self.db.execute(
                "INSERT INTO worlds VALUES (?,?)", (version, json.dumps(beliefs))
            )

    def observations(self):
        return [
            Observation.model_validate_json(r[0])
            for r in self.db.execute("SELECT data FROM observations ORDER BY id")
        ]

    def world(self, version):
        row = self.db.execute(
            "SELECT data FROM worlds WHERE version=?", (version,)
        ).fetchone()
        if row is None:
            raise KeyError(version)
        return json.loads(row[0])

    def close(self):
        self.db.close()


def associate(before, after, ambiguity_margin=0.05, max_cost=0.5):
    """Appearance+class assignment retains competing candidates, no invented probability."""
    if not before or not after:
        return []
    costs = np.ones((len(before), len(after))) * 10
    for i, a in enumerate(before):
        for j, b in enumerate(after):
            if a.label != b.label or a.descriptor is None or b.descriptor is None:
                continue
            va = np.asarray(a.descriptor)
            vb = np.asarray(b.descriptor)
            costs[i, j] = 1 - float(
                va @ vb / (np.linalg.norm(va) * np.linalg.norm(vb) + 1e-12)
            )
    rows, cols = linear_sum_assignment(costs)
    result = []
    for i, j in zip(rows, cols):
        if costs[i, j] > max_cost:
            continue
        alternatives = [
            int(k)
            for k in range(len(after))
            if costs[i, k] <= costs[i, j] + ambiguity_margin
        ]
        # Column ambiguity also matters for repeated identical before objects.
        rivals = [
            int(k)
            for k in range(len(before))
            if costs[k, j] <= costs[i, j] + ambiguity_margin
        ]
        result.append(
            {
                "before": int(i),
                "after": int(j),
                "cost": float(costs[i, j]),
                "alternatives": alternatives,
                "rivals": rivals,
                "identity_verified": len(alternatives) == 1 and len(rivals) == 1,
            }
        )
    return result


def scene_graph(objects, near_distance=0.5):
    edges = []
    for i, a in enumerate(objects):
        for b in objects[i + 1 :]:
            if a.position is None or b.position is None:
                continue
            distance = float(np.linalg.norm(np.asarray(a.position) - b.position))
            if distance < near_distance:
                edges.append(
                    {
                        "from": a.id,
                        "to": b.id,
                        "relation": "NEAR",
                        "distance": distance,
                        "units": "reconstruction units",
                    }
                )
    return {
        "coordinate_convention": "right-handed reconstruction coordinates; no room axes inferred",
        "nodes": [
            {"id": o.id, "label": o.label, "position": o.position} for o in objects
        ],
        "edges": edges,
    }
