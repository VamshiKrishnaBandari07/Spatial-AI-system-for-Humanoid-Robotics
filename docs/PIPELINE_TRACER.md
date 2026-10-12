# Evidence tracer

`PipelineTracer` records PASS, WARN, FAIL and NOT_RUN for specific evidence
checks. It never decides physical change classes or changes model thresholds.
Warnings appear in `summarize_failures()`; an empty audit is NOT_RUN.

```python
from pathlib import Path
from spatialmemory.tracing import PipelineTracer

tracer = PipelineTracer(target_names=["green_mug", "perfume_bottle"])
tracer.audit_crop_and_mask("OBJ_002", 12, [100, 200, 150, 280],
                           sam_mask, .45, image=native_bgr_frame)
tracer.audit_3d_localization("OBJ_001", 12, 4.86)
tracer.summarize_failures()
tracer.save(Path("diagnostics/manual_trace"))
```

The example values are caller inputs, not new model measurements. Masks must
have full native image dimensions. Coverage is computed within the clipped
box and leakage outside it separately. Raw confidence remains uncalibrated.
Geometry accepts only finite nonnegative residuals strictly below the limit.
Appearance diagnostics require valid embeddings and match/competitor evidence.
Removal visibility checks require independently verified distinct view IDs,
patch support and at least two views; a numeric confidence alone cannot pass.

Semantic pipeline runs automatically export `diagnostics/tracer/trace.json`
and numbered native mask-overlay crops. `report.json` links this trace.
For an existing run, preserve its outputs and export to a new directory:

```python
from spatialmemory.tracing import export_run_trace
export_run_trace(Path("results/existing_run"), Path("results/existing_run_trace"))
```

Replay uses saved observations and visibility evidence, not new model calls.
Identity records point to joint association diagnostics; they do not invent
missing competitor distances. Missing per-point residuals remain NOT_RUN.
Export indices enumerate saved observations; original observation IDs are
retained. Visibility records use frame index -1 for multi-view evidence.
Detailed files contain private scene images and must remain outside Git.
