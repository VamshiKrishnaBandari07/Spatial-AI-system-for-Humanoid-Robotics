"""Provenance-gated real-data benchmarks. No inference output becomes ground truth."""

import hashlib
import json
import subprocess
from collections import Counter
from pathlib import Path
from typing import Literal
import numpy as np
from pydantic import Field
from .inputs import Strict, Annotations, load_manifest, resolve
from .evaluation import classification_metrics


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def probe_video(path):
    import cv2

    result = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-show_streams",
            "-show_format",
            "-of",
            "json",
            str(path),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    metadata = json.loads(result.stdout)
    cap = cv2.VideoCapture(str(path))
    try:
        ok, frame = cap.read()
        if not ok or frame is None:
            raise ValueError("Recording has no decodable RGB frames")
    finally:
        cap.release()
    return metadata


class PairEvidence(Strict):
    dataset: str = Field(min_length=1)
    before_recording_id: str = Field(min_length=1)
    after_recording_id: str = Field(min_length=1)
    source_url: str = Field(min_length=1)
    relationship_file: str
    relationship_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    relationship_format: Literal["3RScan", "manual_capture", "unverified"]
    data_origin: Literal["REAL", "SYNTHETIC"]
    physical_changes_documented: bool = False
    documentation_note: str = Field(min_length=1)


def validate_pair(manifest_path, evidence, root, annotations=None, report=None):
    manifest, resolved, warnings = load_manifest(manifest_path, root)
    if (
        manifest.dataset
        and manifest.dataset.get("name")
        and manifest.dataset["name"].casefold() != evidence.dataset.casefold()
    ):
        raise ValueError("Pair dataset does not match the manifest")
    if (
        manifest.dataset
        and manifest.dataset.get("data_origin") == "SYNTHETIC"
        and evidence.data_origin == "REAL"
    ):
        raise ValueError("Synthetic manifest cannot enter a real-data benchmark")
    relation_path = resolve(evidence.relationship_file, root)
    if sha256(relation_path) != evidence.relationship_sha256:
        raise ValueError("Pair relationship evidence checksum mismatch")
    relationship = json.loads(relation_path.read_text())
    before, after = evidence.before_recording_id, evidence.after_recording_id
    same_environment = False
    source_changes = False
    if before != after:
        if evidence.relationship_format == "3RScan":
            if not isinstance(relationship, list):
                raise ValueError("3RScan relationship metadata must be an array")
            for item in relationship:
                if item.get("reference") != before:
                    continue
                for scan in item.get("scans", []):
                    if scan.get("reference") != after:
                        continue
                    same_environment = True
                    source_changes = bool(scan.get("removed") or scan.get("nonrigid"))
                    for entry in scan.get("rigid", []):
                        transform = np.asarray(entry.get("transform", []), dtype=float)
                        if (
                            transform.size == 16
                            and np.isfinite(transform).all()
                            and not np.allclose(transform.reshape(4, 4), np.eye(4))
                        ):
                            source_changes = True
        elif evidence.relationship_format == "manual_capture":
            if not isinstance(relationship, dict):
                raise ValueError("Manual capture metadata must be an object")
            same_environment = (
                relationship.get("before_recording_id") == before
                and relationship.get("after_recording_id") == after
                and relationship.get("same_environment_verified") is True
                and bool(relationship.get("capture_location_evidence"))
                and bool(relationship.get("verified_by"))
            )
            source_changes = bool(relationship.get("documented_changes"))
    videos = [resolved[state]["video"] for state in ["before", "after"]]
    hashes = [sha256(video) for video in videos]
    probes = [probe_video(video) for video in videos]
    if hashes[0] == hashes[1]:
        same_environment = False
        warnings.append(
            "Identical recordings are loader/no-change controls, not documented real-change pairs"
        )
    registration = report.get("registration", {}) if report else {}
    input_match = False
    if report:
        input_match = (
            report.get("scene_id") == manifest.scene_id
            and [i.get("sha256") for i in report.get("inputs", [])] == hashes
        )
        if not input_match:
            warnings.append(
                "Inference report does not match the scene and input hashes"
            )
    aligned = (
        input_match
        and registration.get("valid") is True
        and registration.get("heldout_count", 0) > 0
        and registration.get("heldout_inliers", 0) > 0
    )
    annotation_ready = bool(
        annotations
        and annotations.scene_id == manifest.scene_id
        and annotations.data_origin == "REAL"
        and annotations.provenance
        and annotations.changes
    )
    if annotation_ready:
        annotation_ready = all(
            a.annotation_quality == "INDEPENDENTLY_REVIEWED"
            and a.generation_method
            in ["human", "dataset_native", "model_suggestion_verified"]
            for a in annotations.changes
        )
        annotation_ready = annotation_ready and any(
            a.change_type.value in ["ADDED", "REMOVED", "MOVED"]
            for a in annotations.changes
        )
        for annotation in annotations.changes:
            for observation, identifier in [
                (annotation.before_observation, before),
                (annotation.after_observation, after),
            ]:
                if observation is not None and observation.recording_id != identifier:
                    annotation_ready = False
                    warnings.append(
                        "Annotation observation recording ID does not match the pair"
                    )
    documented = evidence.physical_changes_documented and source_changes
    ready = bool(
        same_environment
        and evidence.data_origin == "REAL"
        and documented
        and aligned
        and annotation_ready
        and report.get("status") == "complete"
    )
    if not aligned:
        warnings.append(
            "Shared registration/static overlap not verified on matching inference output"
        )
    if not annotation_ready:
        warnings.append("Independent reviewed real change annotations are unavailable")
    if not documented:
        warnings.append("Physical changes are not documented")
    return {
        "scene_id": manifest.scene_id,
        "before_recording_id": before,
        "after_recording_id": after,
        "source_dataset": evidence.dataset,
        "split": manifest.dataset.get("split", "not specified")
        if manifest.dataset
        else "not specified",
        "source_url": evidence.source_url,
        "data_origin": evidence.data_origin,
        "shared_environment_relationship_verified": same_environment,
        "relationship_evidence": evidence.relationship_file,
        "relationship_sha256": evidence.relationship_sha256,
        "relationship_limitation": "Official metadata or human capture provenance is required; a declaration is not independent physical verification",
        "camera_geometry_availability": {
            state: {
                key: resolved[state].get(key) is not None
                for key in ["camera", "poses", "depth_dir"]
            }
            for state in ["before", "after"]
        },
        "videos_decodable": True,
        "video_probe": probes,
        "input_hashes": hashes,
        "static_landmark_overlap": registration.get("heldout_inliers")
        if input_match
        else None,
        "registration_verified": bool(aligned),
        "registration_residual": registration.get("rmse") if input_match else None,
        "registration_residual_units": "Report reconstruction units; not metres unless source scale is independently verified",
        "registration_uncertainty": None,
        "ground_truth_change_available": bool(annotation_ready),
        "physical_changes_documented": bool(documented),
        "pair_validity_status": "VERIFIED_FOR_EVALUATION"
        if ready
        else "RELATIONSHIP_ONLY"
        if same_environment
        else "REJECTED",
        "warnings": warnings,
    }


class IdentityMapping(Strict):
    scene_id: str
    ground_truth_to_prediction: dict[str, str]
    provenance: str = Field(min_length=1)
    verified_by: str = Field(min_length=1)
    matching_ground_truth: dict[str, str] | None = None


def benchmark(report, annotations: Annotations, pair, mapping: IdentityMapping):
    if pair["pair_validity_status"] != "VERIFIED_FOR_EVALUATION":
        raise ValueError(
            "Real benchmark blocked: verified same-scene pair, physical changes, independent annotations and matching registered output are required"
        )
    if not (
        report["scene_id"]
        == annotations.scene_id
        == pair["scene_id"]
        == mapping.scene_id
    ):
        raise ValueError("Benchmark scene IDs do not match")
    identities = mapping.ground_truth_to_prediction
    if len(set(identities.values())) != len(identities):
        raise ValueError("Identity mapping must be one-to-one")
    truth_ids = {a.object_id for a in annotations.changes}
    if set(identities) - truth_ids:
        raise ValueError("Mapping contains unannotated ground-truth IDs")
    events = report["events"]
    object_ids = [e["object_id"] for e in events if e.get("object_id")]
    if len(object_ids) != len(set(object_ids)):
        raise ValueError("Duplicate predicted object identities")
    predicted = {e["object_id"]: e["change"] for e in events if e.get("object_id")}
    if set(identities.values()) - set(predicted):
        raise ValueError("Mapping references unavailable prediction IDs")
    known = [a for a in annotations.changes if a.change_type.value != "UNKNOWN"]
    truth = [a.change_type.value for a in known]
    guesses = [predicted.get(identities.get(a.object_id), "UNKNOWN") for a in known]
    result = classification_metrics(truth, guesses)
    extras = [
        e
        for e in events
        if not e.get("object_id") or e["object_id"] not in set(identities.values())
    ]
    physical = {"ADDED", "REMOVED", "MOVED"}
    result.update(
        {
            "scope": "Object classification over annotated identities; unannotated detections are separately accounted for",
            "unknown_ground_truth_excluded": len(annotations.changes) - len(known),
            "abstentions": guesses.count("UNKNOWN"),
            "unmapped_predictions": [
                {"id": e["id"], "change": e["change"]} for e in extras
            ],
            "false_positive_rate": None,
            "false_removal_rate": None,
            "object_matching_accuracy": None,
            "runtime_seconds": report.get("runtime_seconds"),
            "peak_memory_mib": report.get("peak_memory_mib"),
            "registration_residual": report["registration"].get("rmse"),
            "calibration_error": None,
            "centroid_localization_error": None,
            "movement_distance_error": None,
            "bounding_box_iou": None,
            "change_region_iou": None,
            "uncertainty_coverage": None,
            "dataset": pair["source_dataset"],
            "scene_id": report["scene_id"],
            "input_modality": "RGB-only reconstruction"
            if report.get("config", {}).get("geometry_backend") in ["sparse", "vggt"]
            else "see report configuration",
            "model_configuration": report.get("config"),
            "annotation_quality": dict(Counter(a.annotation_quality for a in known)),
            "evaluation_split": pair.get("split", "not specified"),
            "identity_mapping_provenance": mapping.provenance,
            "missing_metrics_reason": "Unavailable metrics are null; memory is not recoverable from runtime, and model confidence is not calibrated",
        }
    )
    if annotations.exhaustive_object_inventory:
        negatives = sum(t == "UNCHANGED" for t in truth)
        false_changed = sum(
            t == "UNCHANGED" and p in physical for t, p in zip(truth, guesses)
        )
        result["false_positive_rate"] = false_changed / negatives if negatives else None
        removal_flags = [(t, p) for t, p in zip(truth, guesses) if p == "REMOVED"]
        extra_removed = sum(e["change"] == "REMOVED" for e in extras)
        denominator = len(removal_flags) + extra_removed
        result["false_removal_rate"] = (
            (sum(t != "REMOVED" for t, _ in removal_flags) + extra_removed)
            / denominator
            if denominator
            else None
        )
        tp = sum(t in physical and p == t for t, p in zip(truth, guesses))
        proposed = sum(p in physical for p in guesses) + sum(
            e["change"] in physical for e in extras
        )
        positive = sum(t in physical for t in truth)
        result["physical_change_precision"] = tp / proposed if proposed else None
        result["physical_change_recall"] = tp / positive if positive else None
        result["physical_change_f1"] = (
            2 * tp / (proposed + positive) if proposed + positive else None
        )
        result["false_removal_definition"] = (
            "Incorrect REMOVED proposals / all REMOVED proposals (false discovery fraction)"
        )
        result["false_positive_definition"] = (
            "Verified UNCHANGED objects predicted ADDED/REMOVED/MOVED / all verified UNCHANGED objects; UNKNOWN predictions are abstentions"
        )
    if mapping.matching_ground_truth and isinstance(
        report.get("predicted_object_associations"), dict
    ):
        pairs = list(mapping.matching_ground_truth.items())
        result["object_matching_accuracy"] = sum(
            report["predicted_object_associations"].get(k) == v for k, v in pairs
        ) / len(pairs)
    return result
