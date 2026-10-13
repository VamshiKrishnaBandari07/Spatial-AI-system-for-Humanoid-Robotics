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

The default input uses two before frames and selects after views using SIFT
correspondences and RANSAC homography support from up to 16 candidates. This
selects comparable image structure, not proof of object identity or empty space.
If matching fails, uniform sampling is explicitly recorded as a fallback.
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
