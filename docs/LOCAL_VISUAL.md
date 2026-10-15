# Local vision-language comparison

The FastAPI upload workspace now supports an optional local multi-image model.
No API account or key is required. Downloading weights requires network access;
inference explicitly uses `local_files_only=True` and does not upload images.

## Setup

Use the existing neural environment (PyTorch 2.6.0, Transformers 4.49.0, Pillow).

```bash
python scripts/install_local_visual.py
SPATIALMEMORY_VISUAL_BACKEND=local python -m uvicorn spatialmemory.app:app --host 127.0.0.1 --port 8000
```

The installer downloads about 1.02 GB of safetensors plus tokenizer/processor
files under `/workspace/spatialmemory-models`. It verifies the checkpoint SHA256.
The pinned model is `HuggingFaceTB/SmolVLM-500M-Instruct`, revision
`a7da5b986cb59b408707209984f360a5f4ad7e47`, Apache-2.0. Inference uses CPU float32.
Allow additional RAM for image processing and generation. There is no arbitrary
remote Python execution. The existing single-user server is not a hardened
public multi-user deployment.

## Use

After loading a completed comparison in the upload UI, click **Run visual
comparison**. A separate panel displays the generated narrative, distinct from
3D change decisions. One local comparison runs at a time. Duplicate requests
for a job are rejected to preserve evidence. On failure, inspect the saved
diagnostics rather than interpreting missing output as no change.

Command-line equivalent:

```bash
OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 python -m spatialmemory.local_visual \
  --run results/room_01_neural --output results/room_01_visual_local
```

The default input selects two pairs using SIFT correspondences and RANSAC
homography support from up to 16 candidates per recording. Each map must cover
at least 70% of a uniform image grid in both directions. This is a conservative
view-selection setting, not a calibrated visibility threshold or proof of object
identity/empty space. Singular maps and projective poles are rejected. If too few
pairs pass, the stage records a view-selection failure and does not run the VLM.
Full-frame homographies can reject useful nonplanar or partial object views;
object-centric view retrieval remains necessary for robust cluttered scenes.
Original timestamps, roles and image hashes are retained. Use
`--frames-per-video 4` for more views at greater compute cost. Sampling is not
exhaustive and can miss short observations. No known-change answer key is fed to
the model. Output is capped at 256 tokens by default; truncation is recorded.

## Interpretation

`visual_changes.json` preserves the model's actual narrative. Its `changes` array
is empty because free text is not automatically converted into validated event
labels. A narrative is not calibrated confidence, proof of removal, a 3D location,
or ground truth. A small VLM may invent objects and miss obvious changes. Running
successfully does not establish useful accuracy. The local model is a candidate
backend for evaluation, not a validated replacement for geometric reasoning.

The GitHub Pages report viewer accepts this output, but Pages cannot execute the
model. The new control is in the FastAPI dashboard; the separate Colab Gradio UI
has not been wired to it. The API adapter remains optional and is not needed for
local operation.

## Measured results

Two real CPU runs completed in about 123 seconds each. Both missed the visible
handled-container removal; the matched-view run instead claimed a lotion bottle
was absent. This checkpoint is not validated for reliable change detection.
See [measured local results](../reports/local_visual_validation.json).

The coverage-gated rerun rejected this phone pair before generation. The earlier
matcher selected a frame cutting off the cabinet-top objects, with only 51.2%
projected source coverage. This repairs misleading view selection; it does not
establish successful object change detection.

## Structured visual findings

The server's visual-comparison action now requests structured output from the
local model. This mode samples native frames in physical timestamp order from
both recording manifests. It does not require a single planar homography to
cover the whole scene; view correspondence remains explicitly unverified and
must be considered by the visual model. This selection is not visibility proof.

```bash
OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 python -m spatialmemory.local_visual \
  --run /path/to/completed/run --output /path/to/new/visual-output \
  --structured --frames-per-video 2 --max-new-tokens 512
```

Model-generated JSON must satisfy the existing visual comparison schema and cite
only supplied frame IDs in their correct before/after roles. Truncated output,
prose, malformed JSON, or invented references cannot become change proposals.
The raw narrative is preserved for diagnosis, and the UI explicitly reports when
no structured output was accepted. Valid JSON remains an unverified visual model
proposal: schema validity does not establish semantic correctness, calibrated
confidence, or a valid 3D location. Manual review is separate and never included
in the model request.

The legacy command without `--structured` retains matched-planar-view selection
and narrative-only output. The structured path enables change proposals but does
not guarantee that the small local checkpoint identifies the actual changes.

### Larger pinned local model

An optional `2b` profile uses `HuggingFaceTB/SmolVLM-Instruct` revision
`81cd9a775a4d644f2faf4e7becff4559b46b14c7` (Apache-2.0). Its checkpoint is
4,492,630,912 bytes, SHA-256
`8a4f76cb64f6f2e4e74716d8fc1cfc6a70bbb3eeea69d424c3ec9902655065eb`.
The model is optional; plan for checkpoint storage and float32 CPU working memory.
The installer verifies the checkpoint hash. Runtime uses pinned local files only.

```bash
python scripts/install_local_visual.py --model-profile 2b
python -m spatialmemory.local_visual --run /path/to/run \
  --output /path/to/new/output --structured --model-profile 2b \
  --frames-per-video 2 --max-new-tokens 512
```

To select this profile for the server, set
`SPATIALMEMORY_VISUAL_MODEL_PROFILE=2b` along with
`SPATIALMEMORY_VISUAL_BACKEND=local` and restart your server. The default remains
`500m`; a larger checkpoint is not a guarantee of accurate change detection.

### Measured structured-output result

The actual four-frame test failed for both checkpoints. The 500M model returned
only a frame heading in 130.97 seconds. The larger model, using eager vision
attention and SDPA decoder attention, returned invented frame IDs and an invalid
ADDED entry in 359.05 seconds. Validation accepted zero change findings.
See `reports/local_structured_visual_validation.json`.

An earlier larger-model eager-attention attempt was interrupted after more than
ten minutes without output. All-SDPA loading failed because the installed vision
encoder lacks that implementation; the final configuration selects each
component separately. These are measured limitations, not successful detection.
The larger profile remains optional and is not the server default.
