"""Scalar temperature fitting for binary detection-correctness logits.

This calibrates detection correctness only, not physical change probability.
Fit on labelled validation scenes; report performance on separate scenes.
"""

from dataclasses import dataclass
import numpy as np
from numpy.typing import NDArray
from scipy.optimize import minimize_scalar
from scipy.special import expit


@dataclass(frozen=True)
class DetectorTemperature:
    temperature: float
    source: str
    model_revision: str

    def __post_init__(self) -> None:
        if (not np.isfinite(self.temperature) or self.temperature <= 0
                or not self.source or not self.model_revision):
            raise ValueError("Positive temperature and calibration provenance required")

    @classmethod
    def fit(cls, logits: NDArray[np.float64], labels: NDArray[np.float64],
            source: str, model_revision: str) -> "DetectorTemperature":
        """Minimize binary NLL over positive T; logits/labels both have shape (N,)."""
        x, y = np.asarray(logits, float), np.asarray(labels, float)
        if (x.ndim != 1 or x.shape != y.shape or len(x) < 2
                or not np.isfinite(x).all() or not np.isfinite(y).all()
                or set(np.unique(y)) != {0.0, 1.0} or not source or not model_revision):
            raise ValueError("Finite logits, both correctness labels and provenance required")

        def nll(log_t: float) -> float:
            scaled = x / np.exp(log_t)
            return float(np.mean(np.logaddexp(0, scaled) - y * scaled))

        result = minimize_scalar(nll, bounds=(-5, 5), method="bounded")
        if not result.success:
            raise ValueError("Temperature fitting failed")
        return cls(float(np.exp(result.x)), source, model_revision)

    def probabilities(self, logits: NDArray[np.float64], model_revision: str) -> NDArray[np.float64]:
        if (model_revision != self.model_revision or not np.isfinite(self.temperature)
                or self.temperature <= 0):
            raise ValueError("Invalid calibration or detector revision mismatch")
        values = np.asarray(logits, float)
        if not np.isfinite(values).all():
            raise ValueError("Nonfinite detector logits")
        return expit(values / self.temperature)
