"""Explicit sensor likelihoods; correlation groups contribute only once."""

import math
from dataclasses import dataclass


@dataclass
class Evidence:
    group: str
    log_likelihood_ratio: float
    source: str


def fuse(evidence, prior=0.5):
    if not 0 < prior < 1:
        raise ValueError("Prior must lie strictly between zero and one")
    groups = {}
    for e in evidence:
        if not math.isfinite(e.log_likelihood_ratio):
            raise ValueError("Invalid likelihood")
        groups.setdefault(e.group, []).append(e)
    contributions = {
        k: sum(e.log_likelihood_ratio for e in v) / len(v) for k, v in groups.items()
    }
    log_odds = math.log(prior / (1 - prior)) + sum(contributions.values())
    posterior = 1 / (1 + math.exp(-max(-700, min(700, log_odds))))
    return {
        "posterior_under_supplied_likelihoods": posterior,
        "calibrated": False,
        "groups": contributions,
        "contradiction": any(v > 0 for v in contributions.values())
        and any(v < 0 for v in contributions.values()),
    }


def rank_views(candidates, seen, budget):
    """Heuristic novelty/quality ranking; not a measured information gain."""
    if budget <= 0:
        return []
    selected = []
    for c in sorted(candidates, key=lambda x: x["quality"], reverse=True):
        if c["viewpoint"] not in seen:
            selected.append(c)
            seen = seen | {c["viewpoint"]}
        if len(selected) >= budget:
            break
    return selected
