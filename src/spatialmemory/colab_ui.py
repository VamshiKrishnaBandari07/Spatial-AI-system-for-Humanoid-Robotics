"""Temporary, single-user Gradio workspace using the unchanged core pipeline."""

from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from queue import Empty, Queue
import shutil
from threading import Lock
from uuid import uuid4
import zipfile

from .contracts import Config
from .pipeline import run

MAX_UPLOAD_BYTES = 512 * 1024 * 1024
ACTIVE = Lock()


def results_zip(output):
    """Export computed results only; exclude uploads, checkpoints and external links."""
    output = Path(output).resolve()
    destination = output.parent / (
        "results.zip" if output.name == "output" else f"{output.name}-results.zip"
    )
    files = [
        output / name
        for name in (
            "report.json",
            "world.npz",
            "progress.json",
            "spatial_memory.json",
            "before_model_outputs.json",
            "after_model_outputs.json",
        )
    ]
    for directory in (
        "reconstruction",
        "changes",
        "reports",
        "diagnostics",
        "evidence",
        "visualization",
        "before",
        "after",
    ):
        files.extend((output / directory).rglob("*"))
    with zipfile.ZipFile(destination, "w", zipfile.ZIP_DEFLATED) as bundle:
        for path in files:
            if (
                path.is_file()
                and path.suffix
                in (".json", ".jsonl", ".npz", ".ply", ".html", ".jpg", ".png")
                and path.resolve().is_relative_to(output)
            ):
                bundle.write(path, path.relative_to(output).as_posix())
    return str(destination)


def analyze(before, after, profile, root, config_path, temporal_tracking=False):
    """Stream real stage updates; each comparison gets an immutable fresh directory."""
    for name, value in (("Before", before), ("After", after)):
        if not value:
            raise ValueError(f"{name} video is required")
        path = Path(value)
        if not path.is_file() or path.suffix.lower() not in (".mp4", ".mov", ".mkv"):
            raise ValueError(f"{name}: select an MP4, MOV or MKV file")
        if not 0 < path.stat().st_size <= MAX_UPLOAD_BYTES:
            raise ValueError(f"{name}: file must be nonempty and at most 512 MiB")
    if not ACTIVE.acquire(blocking=False):
        raise RuntimeError("A comparison is already running. Wait for it to finish.")
    try:
        config = Config.model_validate_json(Path(config_path).read_text())
        if profile == "Geometry diagnostic":
            config = config.model_copy(
                update={
                    "geometry_backend": "sparse",
                    "semantics_enabled": False,
                }
            )
        elif profile != "Neural + objects":
            raise ValueError("Unknown processing profile")
        if temporal_tracking and not config.semantics_enabled:
            raise ValueError("Frame tracking requires Neural + objects analysis")
        config = config.model_copy(
            update={"temporal_tracking_enabled": temporal_tracking}
        )
        job = Path(root) / uuid4().hex
        inputs = job / "input"
        inputs.mkdir(parents=True)
        copies = []
        for state, source in (("before", before), ("after", after)):
            target = inputs / (state + Path(source).suffix.lower())
            shutil.copyfile(source, target)
            copies.append(target)
        output = job / "output"
        updates = Queue()
        with ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(
                run, *copies, output, config, updates.put, scene_id=job.name
            )
            # Clearing old files prevents stale results being shown for a new run.
            yield "Starting comparison…", None, None
            while not future.done():
                try:
                    stage = updates.get(timeout=1)
                    yield f"Processing: {stage.replace('_', ' ')}", None, None
                except Empty:
                    pass
            try:
                report = future.result()
            except Exception as error:
                failure = output / "failure.json"
                yield (
                    f"Comparison failed: {error}. No successful result is claimed.",
                    None,
                    [str(failure)] if failure.is_file() else None,
                )
                return
        counts = dict(Counter(event["change"] for event in report["events"]))
        summary = {
            "scene_id": report["scene_id"],
            "runtime_seconds": report["runtime_seconds"],
            "change_counts": counts,
            "registration": report["registration"],
            "limitations": report["limitations"],
            "accuracy": "Unavailable without independent verified annotations",
            "temporal_tracking": report.get("temporal_tracking"),
        }
        files = [
            str(output / "report.json"),
            results_zip(output),
            str(output / "visualization/index.html"),
        ]
        files.extend(
            str(path)
            for path in (
                output / "visualization/before_temporal.html",
                output / "visualization/after_temporal.html",
            )
            if path.is_file()
        )
        yield (
            "Processing completed. UNKNOWN means insufficient evidence. "
            "Download the ZIP before ending this Colab session.",
            summary,
            files,
        )
    finally:
        ACTIVE.release()


def build_demo(root, config_path):
    import gradio as gr

    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    with gr.Blocks(title="SpatialMemory — Scene Comparison") as demo:
        gr.Markdown("# SpatialMemory\nCompare two recordings of the same indoor area.")
        gr.Markdown(
            "Research preview · CPU inference · Uncalibrated monocular scale. "
            "Unknown results need more evidence. Model scores are not verified accuracy."
        )
        with gr.Row():
            before = gr.File(
                label="Before recording", file_types=[".mp4", ".mov", ".mkv"]
            )
            after = gr.File(
                label="After recording", file_types=[".mp4", ".mov", ".mkv"]
            )
        with gr.Row():
            before_preview = gr.Video(
                label="Before preview — verify recording order", interactive=False
            )
            after_preview = gr.Video(
                label="After preview — verify recording order", interactive=False
            )
        before.change(lambda path: path, before, before_preview, queue=False)
        after.change(lambda path: path, after, after_preview, queue=False)
        gr.Markdown(
            "Check that the Before preview shows the scene before your changes. Analysis uses the original uploaded files."
        )
        profile = gr.Radio(
            ["Neural + objects", "Geometry diagnostic"],
            value="Neural + objects",
            label="Analysis",
            info="Geometry diagnostic does not run neural object models.",
        )
        start = gr.Button("Compare recordings", variant="primary")
        temporal = gr.Checkbox(
            label="Track every frame with neural re-detection",
            value=False,
            info="Additional CPU processing; temporal links are not verified change labels.",
        )
        status = gr.Textbox(label="Processing status", interactive=False)
        summary = gr.JSON(label="Computed results")
        downloads = gr.File(
            label="Report, results ZIP and 3D viewer", file_count="multiple"
        )
        gr.Markdown(
            "Download and extract results.zip, then open visualization/index.html "
            "for the interactive 3D scene and supporting evidence. "
            "Raw recordings are excluded from the ZIP."
        )

        def submit(before, after, profile, temporal):
            yield "Validating recordings and processing options…", None, None
            try:
                yield from analyze(before, after, profile, root, config_path, temporal)
            except (ValueError, RuntimeError) as error:
                yield f"Cannot start comparison: {error}", None, None

        start.click(
            submit,
            [before, after, profile, temporal],
            [status, summary, downloads],
            concurrency_limit=1,
            concurrency_id="comparison",
        )
    return demo.queue(max_size=2, default_concurrency_limit=1)
