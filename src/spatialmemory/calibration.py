"""Likelihood estimation from supplied labeled evidence; no invented calibration."""

from dataclasses import dataclass
import math
from scipy.stats import beta


@dataclass
class LikelihoodCalibration:
    source: str
    rates: dict
    training_groups: int
    heldout_verified: bool = False

    @classmethod
    def fit(cls, records, source, alpha=1.0, beta_prior=1.0):
        if not source or not records:
            raise ValueError("Labeled records and provenance are required")
        unique = {}
        for row in records:
            key = (row["hypothesis_id"], row["correlation_group"], row["signal"])
            if key in unique and unique[key] != row:
                raise ValueError("Conflicting labels in one correlation group")
            unique[key] = row
        rates = {}
        for signal in {r["signal"] for r in unique.values()}:
            result = {}
            for truth in [False, True]:
                rows = [
                    r
                    for r in unique.values()
                    if r["signal"] == signal and r["change_present"] == truth
                ]
                if not rows:
                    raise ValueError(
                        f"{signal} requires changed and unchanged training groups"
                    )
                positives = sum(bool(r["signal_present"]) for r in rows)
                a = alpha + positives
                b = beta_prior + len(rows) - positives
                result[str(truth)] = {
                    "mean": a / (a + b),
                    "beta_parameters": [a, b],
                    "credible_interval_95": beta.ppf([0.025, 0.975], a, b).tolist(),
                    "groups": len(rows),
                }
            rates[signal] = result
        return cls(source, rates, len(unique))

    def likelihood_ratio(self, signal, present):
        if signal not in self.rates:
            raise KeyError(signal)
        p = self.rates[signal]["True"]["mean"]
        q = self.rates[signal]["False"]["mean"]
        return math.log(p / q) if present else math.log((1 - p) / (1 - q))

    def audit(self):
        return {
            "source": self.source,
            "rates": self.rates,
            "training_groups": self.training_groups,
            "heldout_verified": self.heldout_verified,
            "method": "Beta-Bernoulli rates fitted on labeled independent groups",
            "calibrated_for_deployment": self.heldout_verified,
        }
