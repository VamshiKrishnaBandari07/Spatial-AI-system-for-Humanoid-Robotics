"""Local upload application; bounded uploads and job-scoped evidence access."""

import json, os, uuid, threading
import logging
import asyncio
from importlib.util import find_spec
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import numpy as np
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from typing import Literal
from .contracts import Config
from .inputs import Annotations
from fastapi.responses import HTMLResponse, FileResponse, StreamingResponse
from .pipeline import run

app = FastAPI(title="SpatialMemory-X Ultra")
ROOT = Path(
    os.environ.get("SPATIALMEMORY_RUNS", "/workspace/spatialmemory-runs")
).resolve()
ROOT.mkdir(parents=True, exist_ok=True)
POOL = ThreadPoolExecutor(max_workers=1)
LOCK = threading.Lock()
JOBS = {}
MAX_UPLOAD = 512 * 1024 * 1024
LOGGER = logging.getLogger(__name__)


def upload_config(geometry_backend, semantics_enabled):
    """Use the conservative phone workflow without implying calibrated uncertainty."""
    return Config(
        geometry_backend=geometry_backend,
        semantics_enabled=semantics_enabled,
        neural_frames_per_video=4,
        neural_frame_selection="quality_diverse",
        registration_min_heldout=4,
        semantic_max_frames=6,
        association_mode="multiview",
        movement_mode="bounded",
    )


@app.get("/", response_class=HTMLResponse)
def home():
    return (Path(__file__).parent / "dashboard.html").read_text()


@app.get("/health")
def health():
    return {"status": "ok", "capabilities": neural_capabilities()}


@app.get("/annotate", response_class=HTMLResponse)
def annotation_workspace():
    return (Path(__file__).parent / "annotation.html").read_text()


@app.post("/annotations/validate")
def validate_annotations(annotations: "Annotations"):
    return {
        "status": "valid",
        "annotations": annotations.model_dump(mode="json"),
        "benchmark_ready": False,
        "note": "Schema validation only. Verified scene relationship, independent provenance and registered matching inference output are still required for a benchmark.",
    }


def neural_capabilities():
    packages = all(find_spec(name) is not None for name in ["torch", "transformers"])
    sources = Path("/workspace/spatialmemory-thirdparty")
    cache = Path("/workspace/spatialmemory-models")
    geometry = (
        packages
        and (sources / "vggt/vggt").is_dir()
        and any(cache.glob("models--facebook--VGGT-1B/snapshots/*/model.pt"))
    )
    semantics = (
        packages
        and (sources / "sam2/sam2").is_dir()
        and all(
            any((cache / model / "snapshots").glob("*"))
            for model in [
                "models--facebook--sam2.1-hiera-tiny",
                "models--facebook--dinov2-small",
                "models--IDEA-Research--grounding-dino-tiny",
            ]
        )
    )
    return {"sparse": True, "vggt": bool(geometry), "semantics": bool(semantics)}


def persist_status(folder, state):
    temporary = folder / "job.json.tmp"
    temporary.write_text(json.dumps(state))
    temporary.replace(folder / "job.json")


async def save_upload(upload, target):
    size = 0
    try:
        with target.open("wb") as f:
            while chunk := await upload.read(1024 * 1024):
                size += len(chunk)
                if size > MAX_UPLOAD:
                    raise HTTPException(413, "Video exceeds 512 MiB limit")
                f.write(chunk)
        if size == 0:
            raise HTTPException(400, "Empty video")
    except (Exception, asyncio.CancelledError):
        target.unlink(missing_ok=True)
        raise
    finally:
        await upload.close()


@app.post("/jobs")
async def submit(
    before: UploadFile = File(...),
    after: UploadFile = File(...),
    geometry_backend: Literal["sparse", "vggt"] = Form("sparse"),
    semantics_enabled: bool = Form(False),
):
    capabilities = neural_capabilities()
    if geometry_backend == "vggt" and not capabilities["vggt"]:
        raise HTTPException(
            503,
            "VGGT dependencies/checkpoints are not installed; use sparse geometry or install the neural profile",
        )
    if semantics_enabled and not capabilities["semantics"]:
        raise HTTPException(
            503,
            "Semantic dependencies/checkpoints are not installed; disable object masks or install the neural profile",
        )
    with LOCK:
        if any(j["status"] in ["queued", "running"] for j in JOBS.values()):
            raise HTTPException(429, "A job is already active")
        jid = uuid.uuid4().hex
        JOBS[jid] = {"status": "queued", "stage": "uploading"}
    folder = ROOT / jid
    try:
        folder.mkdir()
        await save_upload(before, folder / "before.mp4")
        await save_upload(after, folder / "after.mp4")
    except (Exception, asyncio.CancelledError) as error:
        # No inference has started: discard only this incomplete upload pair.
        for state in ["before", "after"]:
            try:
                (folder / f"{state}.mp4").unlink(missing_ok=True)
            except OSError:
                LOGGER.exception("Could not remove incomplete upload")
        JOBS[jid] = {
            "status": "failed",
            "stage": "uploading",
            "error": str(error) or "Upload interrupted",
        }
        try:
            persist_status(folder, JOBS[jid])
        except OSError:
            LOGGER.exception("Could not persist failed upload status")
        raise
    finally:
        # Close both multipart files even if saving the first one failed.
        await before.close()
        await after.close()

    def execute():
        JOBS[jid] = {"status": "running", "stage": "starting"}
        try:
            run(
                folder / "before.mp4",
                folder / "after.mp4",
                folder / "output",
                config=upload_config(geometry_backend, semantics_enabled),
                progress=lambda s: JOBS[jid].update(stage=s),
            )
            JOBS[jid].update(status="complete")
        except Exception as e:
            JOBS[jid].update(status="failed", error=str(e))
        finally:
            persist_status(folder, JOBS[jid])

    try:
        POOL.submit(execute)
    except Exception as error:
        JOBS[jid] = {"status": "failed", "stage": "starting", "error": str(error)}
        try:
            persist_status(folder, JOBS[jid])
        except OSError:
            LOGGER.exception("Could not persist worker submission failure")
        raise HTTPException(
            503, "Could not start processing; retry the upload"
        ) from error
    return {"id": jid}


def jobpath(jid):
    if len(jid) != 32 or any(c not in "0123456789abcdef" for c in jid):
        raise HTTPException(404, "Unknown job")
    folder = ROOT / jid
    if not folder.is_dir():
        raise HTTPException(404, "Unknown job")
    return folder


@app.get("/jobs/{jid}")
def status(jid: str):
    folder = jobpath(jid)
    if jid in JOBS:
        return JOBS[jid]
    if (folder / "output/report.json").exists():
        return {"status": "complete", "stage": "restored run"}
    saved = folder / "job.json"
    if saved.exists():
        state = json.loads(saved.read_text())
        if state.get("status") == "failed":
            return state
    failure = folder / "output/failure.json"
    if failure.exists():
        return json.loads(failure.read_text())
    return {
        "status": "interrupted",
        "error": "Process restarted; incomplete job must be resubmitted",
    }


@app.get("/jobs/{jid}/report")
def report(jid: str):
    path = jobpath(jid) / "output/report.json"
    if not path.exists():
        raise HTTPException(409, "Report not ready")
    return json.loads(path.read_text())


@app.get("/jobs/{jid}/world")
def world(jid: str):
    path = jobpath(jid) / "output/world.npz"
    if not path.exists():
        raise HTTPException(409, "World not ready")
    with np.load(path) as data:
        return {
            k: data[k][:: max(1, (len(data[k]) + 14999) // 15000)].tolist()
            for k in data.files
        }


@app.get("/jobs/{jid}/frames/{state}/{filename}")
def frame(jid: str, state: str, filename: str):
    if (
        state not in ["before", "after"]
        or Path(filename).name != filename
        or not filename.endswith(".jpg")
    ):
        raise HTTPException(404)
    path = jobpath(jid) / "output" / state / filename
    if not path.is_file():
        raise HTTPException(404)
    return FileResponse(path)


@app.get("/jobs/{jid}/evidence/{filename}")
def evidence(jid: str, filename: str):
    import re

    if not re.fullmatch(r"[A-Za-z0-9_-]+\.png", filename):
        raise HTTPException(404)
    path = jobpath(jid) / "output/evidence/supporting_frames" / filename
    if not path.is_file():
        raise HTTPException(404)
    return FileResponse(path)


@app.get("/jobs/{jid}/video/{state}")
def video(jid: str, state: str):
    if state not in ["before", "after"]:
        raise HTTPException(404)
    path = jobpath(jid) / (state + ".mp4")
    if not path.is_file():
        raise HTTPException(404)
    return FileResponse(path, media_type="video/mp4")


@app.get("/jobs/{jid}/artifacts/{artifact}")
def artifact(jid: str, artifact: str):
    available = {
        "before": "reconstruction/before.ply",
        "after": "reconstruction/after.ply",
        "aligned": "reconstruction/aligned_scene.ply",
        "changes": "changes/changes.json",
        "regions": "changes/changed_regions.ply",
        "registration": "diagnostics/registration.json",
        "association": "diagnostics/association.json",
    }
    if artifact not in available:
        raise HTTPException(404, "Unknown artifact")
    output = (jobpath(jid) / "output").resolve()
    path = (output / available[artifact]).resolve()
    if not path.is_relative_to(output) or not path.is_file():
        raise HTTPException(404, "Artifact unavailable")
    return FileResponse(path, filename=path.name)


@app.get("/jobs/{jid}/results")
def results(jid: str):
    """Export computed results, without raw recordings or neural checkpoints."""
    import tempfile
    import zipfile

    output = (jobpath(jid) / "output").resolve()
    if not (output / "report.json").is_file():
        raise HTTPException(409, "Results not ready")
    archive = tempfile.TemporaryFile()
    try:
        files = [
            output / name
            for name in [
                "report.json",
                "world.npz",
                "progress.json",
                "spatial_memory.json",
            ]
        ]
        for directory in [
            "reconstruction",
            "changes",
            "reports",
            "diagnostics",
            "evidence",
            "visualization",
            "before",
            "after",
        ]:
            files.extend((output / directory).rglob("*"))
        with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED) as bundle:
            for path in files:
                if not path.is_file() or path.suffix not in [
                    ".json",
                    ".npz",
                    ".ply",
                    ".html",
                    ".jpg",
                    ".png",
                ]:
                    continue
                if not path.resolve().is_relative_to(output):
                    continue
                bundle.write(path, path.relative_to(output).as_posix())
        archive.seek(0)
    except Exception:
        archive.close()
        raise

    def chunks():
        try:
            while chunk := archive.read(1024 * 1024):
                yield chunk
        finally:
            archive.close()

    return StreamingResponse(
        chunks(),
        media_type="application/zip",
        headers={
            "Content-Disposition": f'attachment; filename="spatialmemory-{jid}.zip"'
        },
    )
