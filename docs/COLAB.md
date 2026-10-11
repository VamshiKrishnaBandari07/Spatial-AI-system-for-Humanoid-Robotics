# Temporary upload UI on Google Colab

[Open the notebook](https://colab.research.google.com/github/VamshiKrishnaBandari07/Spatial-AI-system-for-Humanoid-Robotics/blob/main/notebooks/SpatialMemory_Colab.ipynb).

1. Open the notebook and connect to a Colab runtime. The current pipeline uses CPU; selecting a GPU does not accelerate these loaders. A higher-memory runtime helps if available, but free capacity is not guaranteed.
2. Read the research-checkpoint licence notice, set `ACCEPT_RESEARCH_TERMS = True` only if appropriate for your use, and run the setup cell. It installs into an isolated virtual environment, fetches pinned source revisions and downloads pinned checkpoints with the existing integrity checks retained.
3. Run the launch cell and enter a password of at least 12 characters. Open the printed `gradio.live` link and sign in as **reviewer** with that password.
4. Upload separate before and after MP4/MOV/MKV files (up to 512 MiB each), leave **Neural + objects** selected and click **Compare recordings**. Progress comes from actual pipeline stages. Geometry diagnostic is a separate non-neural profile, never an automatic fallback.
5. Save **results.zip** before disconnecting. Extract it and open `visualization/index.html` for the 3D viewer and evidence; also retain the report and diagnostics. Original uploads are not included in this export.

The model architecture, thresholds, appearance floor and bounded movement policy are unchanged. Each run has a fresh scene ID and output directory. Missing movement error budgets intentionally cause abstention. Independent original-video annotations are needed for accuracy; a completed run alone is not a benchmark.

## Hosting and limitations

This produces a temporary authenticated link, not an always-on GitHub website. Google Colab hosts the uploaded recordings; the Gradio sharing service relays browser traffic. Do not share the password. No video is submitted to a remote model inference API. Closing/stopping the runtime ends the service; Colab may terminate idle or resource-heavy sessions. Files under `/content` are ephemeral, and raw uploads stay there until deleted or the runtime ends.

The verified CPU prototype has previously used approximately 10 GiB peak process memory, excluding notebook/server overhead. Installation needs several GB for weights and PyTorch; long videos and concurrent tasks add memory/storage pressure. Out-of-memory termination may prevent a failure report. Use short, slow, overlapping captures initially. More powerful or paid resources may be necessary; free availability is not a guarantee.

Checkpoints retain their licences, including VGGT's noncommercial research restriction. This notebook does not grant commercial rights for a humanoid company evaluation. Consult [MODEL_REGISTRY.md](../MODEL_REGISTRY.md).

## Verification scope

The Gradio application and core tests can be exercised locally. This cloud workspace cannot authenticate to your Google account or establish a Colab runtime on your behalf. Notebook installation on a fresh Colab machine and the public Gradio relay therefore require your first notebook execution. The notebook reports clone revision, memory/storage, installation errors and the live link rather than claiming deployment in advance.
