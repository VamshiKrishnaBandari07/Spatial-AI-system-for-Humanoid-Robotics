import json, time, platform, importlib.metadata
from pathlib import Path
import numpy as np
from .contracts import Config
from .ingestion import ingest, keyframes
from .geometry import reconstruct, bundle_adjust
from .registration import register, transform
from .change import compare
from .execution import locked_run


@locked_run
def run(
    before,
    after,
    output,
    config=None,
    progress=None,
    scene_id="direct_input",
    input_metadata=None,
):
    config = config or Config()
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    if (output / "report.json").exists():
        raise FileExistsError(
            "Choose a new output directory; existing runs are immutable"
        )
    input_metadata = input_metadata or {}
    stages = []
    started = time.perf_counter()

    def stage(name):
        stages.append({"stage": name, "elapsed_seconds": time.perf_counter() - started})
        (output / "progress.json").write_text(json.dumps(stages, indent=2))
        if progress:
            progress(name)

    try:
        stage("ingestion_before")
        bf, bm = ingest(
            Path(before),
            output / "before",
            config,
            "before",
            input_metadata.get("before", {}).get("timestamp_data"),
        )
        stage("ingestion_after")
        af, am = ingest(
            Path(after),
            output / "after",
            config,
            "after",
            input_metadata.get("after", {}).get("timestamp_data"),
        )
        from .inputs import validate_sensor_timing

        validate_sensor_timing(input_metadata.get("before", {}), bm)
        validate_sensor_timing(input_metadata.get("after", {}), am)
        if config.geometry_backend == "vggt":
            from .neural_geometry import (
                load_expert,
                reconstruct_neural,
                reusable_predictions,
            )

            cached = None
            expert = None
            if config.geometry_reuse_from:
                stage("geometry_cache_validation")
                cached = reusable_predictions(
                    config.geometry_reuse_from, config, bm, am
                )
            else:
                stage("neural_model_loading")
                expert = load_expert()
            bk = keyframes(bf)
            ak = keyframes(af)
            bk = [
                bk[i]
                for i in np.linspace(
                    0, len(bk) - 1, min(len(bk), config.neural_frames_per_video)
                )
                .round()
                .astype(int)
            ]
            ak = [
                ak[i]
                for i in np.linspace(
                    0, len(ak) - 1, min(len(ak), config.neural_frames_per_video)
                )
                .round()
                .astype(int)
            ]
            stage("reconstruction_before")
            b = reconstruct_neural(
                bk,
                output / "neural/before",
                expert,
                cached["before"] if cached else None,
            )
            stage("reconstruction_after")
            a = reconstruct_neural(
                ak, output / "neural/after", expert, cached["after"] if cached else None
            )
            del expert
        else:
            stage("reconstruction_before")
            b = bundle_adjust(
                reconstruct(
                    keyframes(bf),
                    config,
                    input_metadata.get("before", {}).get("camera_data"),
                )
            )
            stage("reconstruction_after")
            a = bundle_adjust(
                reconstruct(
                    keyframes(af),
                    config,
                    input_metadata.get("after", {}).get("camera_data"),
                )
            )
        b_observations = []
        a_observations = []
        b_masks = {}
        a_masks = {}
        spatial_memory = None
        semantic_events = []
        if config.semantics_enabled:
            from .semantics import SemanticExperts, observe_scene

            stage("semantic_model_loading")
            semantic_experts = SemanticExperts()
            stage("semantic_before")
            b_observations, b_masks = observe_scene(
                b, bf, "before", output, config, semantic_experts
            )
            stage("semantic_after")
            a_observations, a_masks = observe_scene(
                a, af, "after", output, config, semantic_experts
            )
            del semantic_experts
        stage("shared_registration")
        registration = register(b, a, config, b_masks, a_masks)
        if not registration.valid:
            raise ValueError("Registration gate failed: " + registration.reason)
        aligned = transform(
            a.points,
            registration.scale,
            np.array(registration.rotation),
            np.array(registration.translation),
        )
        stage("geometric_candidates")
        events = compare(b.points, aligned, registration, config)
        if config.semantics_enabled:
            from .semantics import (
                common_frame,
                group_objects,
                consensus,
                persist_memory,
            )

            stage("semantic_consensus")
            a_observations = common_frame(a_observations, registration)
            before_objects = group_objects(b_observations, "before", config)
            after_objects = group_objects(a_observations, "after", config)
            semantic_events, associations = consensus(
                before_objects, after_objects, b, a, registration, config
            )
            spatial_memory = persist_memory(
                output, before_objects, after_objects, semantic_events
            )
            events.extend(semantic_events)
        np.savez_compressed(
            output / "world.npz",
            before=b.points,
            after=aligned,
            before_colors=b.colors,
            after_colors=a.colors,
        )
        cameras = []
        for state, r in [("before", b), ("after", a)]:
            for fid, pose in r.poses.items():
                center = -pose[:3, :3].T @ pose[:3, 3]
                if state == "after":
                    center = transform(
                        center[None],
                        registration.scale,
                        np.array(registration.rotation),
                        np.array(registration.translation),
                    )[0]
                cameras.append(
                    {"frame_id": fid, "state": state, "position": center.tolist()}
                )
        stage("complete")
        report = {
            "scene_id": scene_id,
            "schema_version": "1.0",
            "status": "complete",
            "workflow": "CPU "
            + config.geometry_backend
            + (
                " semantic consensus"
                if config.semantics_enabled
                else " geometry baseline"
            ),
            "config": config.model_dump(),
            "inputs": [bm, am],
            "registration": registration.model_dump(),
            "geometry": {"before": b.diagnostics, "after": a.diagnostics},
            "events": [e.model_dump(mode="json") for e in events],
            "spatial_memory": spatial_memory,
            "cameras": cameras,
            "stages": stages,
            "runtime_seconds": time.perf_counter() - started,
            "hardware": {
                "platform": platform.platform(),
                "python": platform.python_version(),
            },
            "versions": {
                m: importlib.metadata.version(m)
                for m in ["numpy", "scipy", "opencv-python-headless", "pydantic"]
            },
            "optional_inputs": {
                state: {
                    k: (
                        "validated, retained; not fused by sparse baseline"
                        if k in ["depth_dir", "poses", "imu"]
                        else (
                            "validated; VGGT predicts intrinsics, calibration not fused"
                            if k == "camera" and config.geometry_backend == "vggt"
                            else "used"
                        )
                    )
                    for k, v in fields.items()
                    if k in ["camera", "depth_dir", "poses", "imu", "timestamps"]
                    and v is not None
                }
                for state, fields in input_metadata.items()
            },
            "limitations": [
                "Semantic labels and depth are model predictions, not verified ground truth."
                if config.semantics_enabled
                else "Sparse geometric candidates have no semantic identity.",
                "No measured depth: absence is UNKNOWN.",
                "No metric scale calibration.",
                "Static-background dominance is assumed, not guaranteed.",
            ],
        }
        from .exports import export_run

        export_run(output, report, b.points, aligned, b.colors, a.colors)
        # A restart must never interpret a partially written report as a completed run.
        temporary_report = output / "report.json.tmp"
        temporary_report.write_text(json.dumps(report, indent=2))
        temporary_report.replace(output / "report.json")
        return report
    except Exception as e:
        (output / "failure.json").write_text(
            json.dumps(
                {
                    "status": "failed",
                    "stage": stages[-1]["stage"] if stages else None,
                    "error": str(e),
                    "type": type(e).__name__,
                    "runtime_seconds": time.perf_counter() - started,
                },
                indent=2,
            )
        )
        raise
