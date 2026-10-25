import json, logging, time, platform, importlib.metadata
from pathlib import Path
import numpy as np
from .contracts import Config
from .ingestion import ingest, keyframes
from .geometry import reconstruct, bundle_adjust
from .registration import register, transform
from .change import compare
from .execution import locked_run

LOGGER = logging.getLogger(__name__)


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
    if config.temporal_tracking_enabled and not config.semantics_enabled:
        raise ValueError("Temporal tracking requires neural semantic anchor masks")
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
            if config.neural_frame_selection == "quality_diverse":
                from .ingestion import neural_keyframes

                bk = neural_keyframes(bf, config.neural_frames_per_video)
                ak = neural_keyframes(af, config.neural_frames_per_video)
            stage("reconstruction_before")
            b = reconstruct_neural(
                bk,
                output / "neural/before",
                expert,
                cached["before"] if cached else None,
                config.neural_preprocessing,
            )
            stage("reconstruction_after")
            a = reconstruct_neural(
                ak,
                output / "neural/after",
                expert,
                cached["after"] if cached else None,
                config.neural_preprocessing,
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
        if config.metric_geometry_enabled:
            if config.geometry_backend != "vggt" or not config.registration_initial:
                raise ValueError(
                    "Metric neural workflow requires VGGT and an explicit sourced registration initial transform"
                )
            from .sensor_geometry import reconstruct_sensor

            stage("metric_sensor_reconstruction")
            b = reconstruct_sensor(bf, input_metadata.get("before", {}), b.diagnostics)
            a = reconstruct_sensor(af, input_metadata.get("after", {}), a.diagnostics)
        b_observations = []
        a_observations = []
        b_masks = {}
        a_masks = {}
        spatial_memory = None
        semantic_events = []
        if config.semantics_enabled:
            from .semantics import SemanticExperts, observe_scene

            if config.classical_pose_validation_enabled:
                from .classical_validation import validate_classical_cameras

                stage("classical_camera_validation")
                b = validate_classical_cameras(
                    b, bf, output / "diagnostics/before_classical", config
                )
                a = validate_classical_cameras(
                    a, af, output / "diagnostics/after_classical", config
                )
            semantic_experts = None
            if not config.perception_reuse_from or config.temporal_tracking_enabled:
                stage("semantic_model_loading")
                semantic_experts = SemanticExperts(config)
            stage("semantic_before")
            if config.perception_reuse_from:
                from .perception_cache import cached_observations

                b_observations, b_masks = cached_observations(
                    Path(config.perception_reuse_from),
                    "before",
                    b,
                    bf,
                    output,
                    config,
                    bm["sha256"],
                )
            else:
                b_observations, b_masks = observe_scene(
                    b, bf, "before", output, config, semantic_experts
                )
            stage("semantic_after")
            if config.perception_reuse_from:
                a_observations, a_masks = cached_observations(
                    Path(config.perception_reuse_from),
                    "after",
                    a,
                    af,
                    output,
                    config,
                    am["sha256"],
                )
            else:
                a_observations, a_masks = observe_scene(
                    a, af, "after", output, config, semantic_experts
                )
            if not config.temporal_tracking_enabled:
                del semantic_experts
        stage("shared_registration")
        if config.metric_geometry_enabled:
            from .metric_alignment import metric_register
            from .contracts import Registration
            from .benchmarking import sha256

            initial = json.loads(Path(config.registration_initial).read_text())
            if (
                initial.get("unit") != "metres"
                or initial.get("source_frame") != "rescan_world"
                or initial.get("destination_frame") != "reference_world"
            ):
                raise ValueError("Unsupported metric initial-transform convention")
            if sha256(initial["relationship_file"]) != initial["relationship_sha256"]:
                raise ValueError("Source transform relationship checksum mismatch")
            if (
                initial.get("before_video_sha256") != bm["sha256"]
                or initial.get("after_video_sha256") != am["sha256"]
            ):
                raise ValueError(
                    "Metric initial transform belongs to different recordings"
                )
            relationship = json.loads(Path(initial["relationship_file"]).read_text())
            native_pair = next(
                (
                    s
                    for item in relationship
                    if item.get("reference") == initial.get("before_recording_id")
                    for s in item.get("scans", [])
                    if s.get("reference") == initial.get("after_recording_id")
                ),
                None,
            )
            if native_pair is None or not np.allclose(
                np.asarray(native_pair["transform"]).reshape(4, 4, order="F"),
                initial["T_reference_rescan"],
                atol=1e-6,
            ):
                raise ValueError(
                    "Metric seed does not match the native reference/rescan transform"
                )
            metrics = metric_register(
                a.points,
                b.points,
                initial["T_reference_rescan"],
                config.registration_threshold,
            )
            (output / "registration_diagnostics.json").write_text(
                json.dumps(metrics, indent=2)
            )
            T = np.asarray(metrics["T_reference_rescan"])
            registration = Registration(
                valid=metrics["valid"],
                scale=1,
                rotation=T[:3, :3].tolist(),
                translation=T[:3, 3].tolist(),
                inliers=metrics["refined"]["inliers"],
                correspondences=metrics["refined"]["source_points"],
                rmse=metrics["refined"]["inlier_rmse_metres"],
                threshold=config.registration_threshold,
                reason=metrics["reason"],
                heldout_count=metrics["heldout"]["source_points"],
                heldout_inliers=metrics["heldout"]["inliers"],
                heldout_rmse=metrics["heldout"]["inlier_rmse_metres"],
                metric_diagnostics=metrics,
            )
        else:
            registration = register(b, a, config, b_masks, a_masks)
            (output / "registration_diagnostics.json").write_text(
                registration.model_dump_json(indent=2)
            )
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
            before_objects = group_objects(
                b_observations,
                "before",
                config,
                identity_checkpoint=output
                / "diagnostics/before_identity_checkpoint.json",
            )
            after_objects = group_objects(
                a_observations,
                "after",
                config,
                identity_checkpoint=output
                / "diagnostics/after_identity_checkpoint.json",
            )
            if config.association_mode == "joint":
                from .identity_integration import save_identity_audit

                save_identity_audit(output, before_objects, "before")
                save_identity_audit(output, after_objects, "after")
            from .association import match_objects
            from .semantics import visibility_evidence
            from collections import Counter

            candidate_diagnostics = []
            diagnostic_matches = match_objects(
                before_objects, after_objects, config, diagnostics=candidate_diagnostics
            )
            association_audit = {
                "schema_version": "1.0",
                "method": config.association_mode,
                "score_is_probability": False,
                "matching_accuracy": None,
                "ground_truth": "Independent correspondences unavailable",
                "candidates": candidate_diagnostics,
                "assignments": diagnostic_matches,
                "gate_counts": dict(
                    Counter(row["gate"] or "accepted" for row in candidate_diagnostics)
                ),
                "visibility": {
                    "before": {
                        obj.id: visibility_evidence(obj, a, config, registration)
                        for obj in before_objects
                    },
                    "after": {
                        obj.id: visibility_evidence(obj, b, config)
                        for obj in after_objects
                    },
                },
                "limitations": [
                    "Raw mask IoU is a model prediction, not measured mask quality",
                    "Observed extents are partial surfaces, not full object dimensions",
                ],
            }
            (output / "diagnostics").mkdir(exist_ok=True)
            (output / "diagnostics/association.json").write_text(
                json.dumps(association_audit, indent=2)
            )
            semantic_events, associations = consensus(
                before_objects, after_objects, b, a, registration, config
            )
            spatial_memory = persist_memory(
                output, before_objects, after_objects, semantic_events
            )
            events.extend(semantic_events)
        if config.metric_geometry_enabled:
            for event in events:
                event.coordinate_system = "reference native world; right-handed"
                event.scale_status = "verified sensor metres"
                event.displacement_units = "metres"
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
        stage("exporting")
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
            "association_diagnostics": "diagnostics/association.json"
            if config.semantics_enabled
            else None,
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
                        (
                            "used in calibrated metric reconstruction"
                            if config.metric_geometry_enabled
                            and k in ["depth_dir", "poses"]
                            else "validated, retained; not fused by sparse baseline"
                        )
                        if k in ["depth_dir", "poses", "imu"]
                        else (
                            (
                                "used for calibrated RGB/depth projection"
                                if config.metric_geometry_enabled
                                else "validated; VGGT predicts intrinsics, calibration not fused"
                            )
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
                "Missing measured-depth pixels preserve UNKNOWN visibility."
                if config.metric_geometry_enabled
                else "No measured depth: absence is UNKNOWN.",
                "Sensor metric scale used; uncertainty not independently calibrated."
                if config.metric_geometry_enabled
                else "No metric scale calibration.",
                "Static-background dominance is assumed, not guaranteed.",
            ],
        }
        from .exports import export_run

        export_run(output, report, b.points, aligned, b.colors, a.colors)
        if config.semantics_enabled:
            from .object_review import create_object_review

            create_object_review(output)
            if config.temporal_tracking_enabled:
                from .temporal_review import track_recording
                from .temporal_neural import neural_redetector

                stage("temporal_tracking")
                redetect = neural_redetector(semantic_experts, config)

                report["temporal_tracking"] = {
                    "before": track_recording(
                        before, output, "before", redetect=redetect
                    ),
                    "after": track_recording(after, output, "after", redetect=redetect),
                }
        # A restart must never interpret a partially written report as a completed run.
        stage("complete")
        report["runtime_seconds"] = time.perf_counter() - started
        if config.semantics_enabled:
            from .tracing import export_run_trace
            export_run_trace(output)
            report["pipeline_trace"] = "diagnostics/tracer/trace.json"
        temporary_report = output / "report.json.tmp"
        temporary_report.write_text(json.dumps(report, indent=2))
        temporary_report.replace(output / "report.json")
        # Dense room export reads report.json (status=complete); must run after it exists.
        if config.geometry_backend == "vggt":
            try:
                stage("dense_room_visualization")
                from .dense_export import write_dense_world

                write_dense_world(output, stride=3)
            except Exception:
                # Sparse world.npz remains available; dense preview is best-effort.
                LOGGER.exception("Dense room visualization export failed")
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
