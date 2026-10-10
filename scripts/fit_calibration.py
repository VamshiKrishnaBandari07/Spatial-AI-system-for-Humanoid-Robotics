"""Audit a likelihood model fitted from actual supplied labeled training records."""

import argparse, json
from pathlib import Path
from spatialmemory.calibration import LikelihoodCalibration

p = argparse.ArgumentParser()
p.add_argument("input", type=Path)
p.add_argument("--output", type=Path, required=True)
a = p.parse_args()
data = json.loads(a.input.read_text())
fit = LikelihoodCalibration.fit(data["records"], data["source"])
a.output.write_text(json.dumps(fit.audit(), indent=2))
print(
    "Training likelihood audit saved. Held-out deployment calibration is not established."
)
