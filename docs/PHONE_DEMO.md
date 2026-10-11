# Smartphone comparison: recording and validation

The application is a local research prototype. A successful run demonstrates execution; verified annotated changes are required to measure accuracy. A result marked Unknown is an abstention, not an unchanged object.

## Launch

From the repository:

```bash
source /workspace/spatialmemory-venv/bin/activate
python -m uvicorn spatialmemory.app:app --host 127.0.0.1 --port 8000
```

Open the application in a supported local browser. In a hosted environment, use its supported port-forwarding mechanism if available; cloud onboarding has no public localhost preview. Do not start a second server on an occupied port. Backend changes require restarting your application process; unrelated jobs and training must remain untouched. `GET /health` reports locally available model profiles, not verified model accuracy.

## Record

Record two genuine phone videos of the same indoor area, approximately 15–60 seconds each. Walk slowly and show static walls, corners and furniture from overlapping viewpoints. Show each relevant object from several angles, including its old and new locations. Avoid blank-wall close-ups, motion blur and severe lighting changes. Keep the original files; resolution, duration and camera path need not match. H.264 MP4 is recommended for browser preview compatibility. Unsupported browser codecs show a preview message; server ingestion separately validates decoding, and extracted evidence images remain available after processing.

Keep an independently recorded list of actual changes. Include unchanged objects and an occluded object to check abstention. Measure small and larger movements where possible. A separate unchanged capture pair is useful for error characterization, but one control does not establish a calibrated error bound for every scene.

## Analyze and review

1. Select before and after videos, choose **Neural reconstruction**, and enable **Analyze objects**.
2. Start analysis. Stage messages show actual progress; CPU inference can take several minutes. A failed alignment is rejected, not replaced with identity.
3. Inspect both scene layers, alignment diagnostics and object evidence. Review compound labels, duplicate tracks, missing observations and unexpected detections.
4. Read Unknown blocking checks and download object matching diagnostics when available. Raw detector/mask scores are not calibrated confidence. Monocular reconstruction units are not metres.
5. Download the report, review notes and **all results** ZIP. The ZIP contains computed scene artifacts and supporting frames; raw recordings and model checkpoints are excluded. Human review notes are a separate browser export.

Movement remains Unknown without explicit validated error budgets in the bounded workflow. The upload UI does not estimate these budgets. Known-class predictions remain proposals with evidence; no confirmed-change accuracy or production readiness is established.

## Independent annotation

Use `/annotate` or copy [the empty template](templates/phone_changes.json). Replace its scene identifier with the report's `scene_id`, add only independently observed records, and specify original before/after frames or timestamps. Set the data origin to REAL only for actual captures. Keep uncertain changes Unknown. Do not infer removal from absence in one view or invent metric coordinates. Do not mark independent review complete without a distinct reviewer inspecting original evidence.

The template intentionally contains zero objects and no asserted observations. Validate with `POST /annotations/validate`; this checks the schema, not benchmark readiness. Follow [the full protocol](annotation_protocol.md) for provenance, independent review and explicit mapping from ground-truth IDs to prediction IDs. Dashboard assessments are not independent ground truth.
