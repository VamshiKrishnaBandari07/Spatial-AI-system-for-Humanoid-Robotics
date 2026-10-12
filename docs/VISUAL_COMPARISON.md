# Multi-view visual comparison

The optional visual-language stage compares sampled native images from a completed
run. It complements the geometric pipeline by proposing visually apparent changes,
including objects whose 3D localization was inconclusive. It does not turn those
proposals into verified 3D events. The chat assistant itself is not embedded in the
repository.

## Run

Configure `OPENAI_API_KEY` securely in the execution environment. Do not commit it
or paste it into chat. The live command sends up to 16 images by default (eight per
recording) to OpenAI and may incur API charges. Local inference is not implemented
by this adapter. Select a model supporting image inputs and JSON object responses.

```bash
python -m spatialmemory.visual_comparison \
  --run results/room_01_neural \
  --output results/room_01_visual \
  --model gpt-4.1
```

Add `--prepare-only` to select and audit images without making an API call. The
output directory must be new. Frame selection uses original presentation timestamps
within each explicit recording role; identical timestamps across recordings do not
imply matching views. `--frames-per-video` accepts 2–16 (default 8). Sampling can
miss brief observations and is not exhaustive tracking.

## Results and integration boundary

`request.json` records frame IDs, timestamps, hashes, model and prompt.
`response.json` preserves the actual API response. `visual_changes.json` contains
model-proposed ADDED, REMOVED, MOVED, UNCHANGED or INCONCLUSIVE observations, cited
before/after frames, visibility reasoning and uncertainty. Missing credentials,
API errors and invalid evidence references fail explicitly; no result is invented.

The report is a separate downstream artifact of the existing video pipeline. It
is not yet connected to the upload UI or fused into `changes/changes.json`.
References and schema are checked in code; their visual truth is not established
by schema validation. The VLM can hallucinate, confuse lookalikes, or misinterpret
occlusion/reflections. It provides neither calibrated confidence nor measured 3D
positions. Removal means apparent absence at a visible location, not proof that
an object left the room. Independent annotations must remain outside its inputs.

## Validation status

Automated tests cover temporal ordering, path containment, wrong-role/invented
references, offline preparation and missing credentials. These are adapter tests,
not model accuracy tests. A real request was prepared for the uploaded room pair;
no live VLM inference has been run because no API credential is configured.
