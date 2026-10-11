"""Every-frame 2D propagation between real neural anchors, never 3D change proof."""

import json
import time
import html
from pathlib import Path
import cv2
import numpy as np
from .association import semantic_label, unit_descriptor
from .segmentation import propagate_mask
from .object_review import proposed_name


def link_redetections(proposals, active, appearances, names):
    """Reject ambiguity on either side; no greedy identity assignment."""
    candidates = []
    for number, proposal in enumerate(proposals):
        mask = np.asarray(proposal["mask"], bool)
        descriptor = unit_descriptor(proposal["descriptor"])
        for tid, previous_mask in active.items():
            if tid not in appearances or semantic_label(
                names.get(tid, "")
            ) != semantic_label(proposal["label"]):
                continue
            cosine = float(descriptor @ appearances[tid])
            overlap = float(
                (mask & previous_mask).sum() / max(1, (mask | previous_mask).sum())
            )
            if cosine >= 0.85 and overlap >= 0.2:
                candidates.append((0.5 * cosine + 0.5 * overlap, number, tid))
    links = {}
    for score, number, tid in candidates:
        rivals = [
            s
            for s, n, t in candidates
            if (n == number or t == tid) and (n, t) != (number, tid)
        ]
        if not rivals or score - max(rivals) >= 0.05:
            links[number] = tid
    return links


def track_recording(
    video, output, state, width=320, redetect=None, redetect_seconds=2.0
):
    started = time.perf_counter()
    output = Path(output)
    metadata = json.loads((output / state / "manifest.json").read_text())
    timestamps = metadata["original_timestamps_seconds"]
    proposals = json.loads((output / f"{state}_model_outputs.json").read_text())
    anchors = {int(row["frame_id"].split(":")[-1]): row for row in proposals}
    destination = output / "diagnostics" / f"{state}_temporal.jsonl"
    destination.parent.mkdir(parents=True, exist_ok=True)
    capture = cv2.VideoCapture(str(video))
    if not capture.isOpened():
        raise ValueError("Cannot decode tracking video")
    capture.set(cv2.CAP_PROP_ORIENTATION_AUTO, 1)
    active, previous, index = {}, None, 0
    appearances, names = {}, {}
    last_detection = -float("inf")
    redetection_count, identity_links = 0, 0
    pending_redetection = False
    cards = []
    crop_directory = output / "evidence" / "temporal_tracking" / state
    crop_directory.mkdir(parents=True, exist_ok=True)
    try:
        with destination.open("w") as stream:
            while True:
                ok, image = capture.read()
                if not ok:
                    break
                if index >= len(timestamps):
                    raise ValueError("Tracking frames exceed recorded timestamps")
                h, w = image.shape[:2]
                image = cv2.resize(image, (width, max(1, round(h * width / w))))
                states = []
                flow_cache = {}
                if previous is not None:
                    for track_id, mask in list(active.items()):
                        propagated, audit = propagate_mask(
                            previous, image, mask, flow_cache=flow_cache
                        )
                        area_ratio = float(propagated.sum() / max(1, mask.sum()))
                        drift = (
                            audit["requires_reinitialization"]
                            or not 0.5 <= area_ratio <= 2
                        )
                        states.append(
                            {
                                "track_segment_id": track_id,
                                "status": "requires_redetection"
                                if drift
                                else "flow_supported_2d",
                                "area_ratio": area_ratio,
                                "flow": audit,
                            }
                        )
                        if drift:
                            pending_redetection = True
                            del active[track_id]
                        else:
                            active[track_id] = propagated
                if redetect is not None and (
                    index in anchors
                    or timestamps[index] - last_detection >= redetect_seconds
                    or (
                        pending_redetection
                        and timestamps[index] - last_detection >= 0.5
                    )
                ):
                    proposals_now = redetect(image)
                    old_active = active
                    active = {}
                    links = link_redetections(
                        proposals_now, old_active, appearances, names
                    )
                    for number, proposal in enumerate(proposals_now):
                        mask = np.asarray(proposal["mask"], bool)
                        if mask.shape != image.shape[:2]:
                            raise ValueError("Re-detection mask dimensions disagree")
                        descriptor = unit_descriptor(proposal["descriptor"])
                        linked = number in links
                        tid = (
                            links[number]
                            if linked
                            else f"{state}_redetect_{index}_{number}"
                        )
                        active[tid] = mask
                        names[tid] = proposal["label"]
                        appearances[tid] = descriptor
                        identity_links += int(linked)
                        yy, xx = np.nonzero(mask)
                        if not len(xx):
                            raise ValueError("Re-detection returned an empty mask")
                        crop_path = crop_directory / f"{index}_{number}.jpg"
                        if not cv2.imwrite(
                            str(crop_path),
                            image[yy.min() : yy.max() + 1, xx.min() : xx.max() + 1],
                        ):
                            raise ValueError("Could not write temporal evidence crop")
                        crop_relative = crop_path.relative_to(output).as_posix()
                        cards.append(
                            f'<article><h3>{html.escape(proposed_name(proposal["label"]))}</h3><img src="../{crop_relative}" alt="Neural observation crop"><p>Frame {index} · {html.escape(tid)}</p></article>'
                        )
                        states.append(
                            {
                                "track_segment_id": tid,
                                "status": "neural_redetection",
                                "proposed_label": proposal["label"],
                                "identity_link_supported": linked,
                                "identity_is_calibrated": False,
                                "crop_file": crop_relative,
                                "raw_detector_score": proposal.get(
                                    "raw_detector_score"
                                ),
                                "detector_provenance": proposal.get(
                                    "detector_provenance"
                                ),
                                "appearance_provenance": proposal.get(
                                    "appearance_provenance"
                                ),
                                "identity_link_method": "mutual unambiguous appearance and propagated-mask overlap",
                            }
                        )
                    last_detection = timestamps[index]
                    redetection_count += 1
                    pending_redetection = False
                elif index in anchors:
                    # Independent segments: do not invent identity links at anchors.
                    active = {}
                    row = anchors[index]
                    for detection_index, label in enumerate(
                        row["detections"]["labels"]
                    ):
                        track_id = f"{state}_{index}_{detection_index}"
                        path = (
                            output
                            / "evidence/supporting_frames"
                            / f"{track_id}_mask.png"
                        )
                        if not path.is_file():
                            continue
                        mask = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
                        if mask is None:
                            raise ValueError("Unreadable neural anchor mask")
                        if abs(mask.shape[1] / mask.shape[0] - w / h) > 0.02:
                            raise ValueError(
                                "Cached anchor mask uses cropped or padded coordinates; provide full-frame neural re-detection"
                            )
                        mask = (
                            cv2.resize(
                                mask,
                                (image.shape[1], image.shape[0]),
                                interpolation=cv2.INTER_NEAREST,
                            )
                            > 0
                        )
                        active[track_id] = mask
                        states.append(
                            {
                                "track_segment_id": track_id,
                                "status": "neural_anchor",
                                "proposed_label": label,
                                "cross_anchor_identity_verified": False,
                            }
                        )
                stream.write(
                    json.dumps(
                        {
                            "frame_index": index,
                            "timestamp_seconds": timestamps[index],
                            "tracks": states,
                            "physical_change_verified": False,
                        }
                    )
                    + "\n"
                )
                previous = image
                index += 1
    finally:
        capture.release()
    if index != len(timestamps):
        raise ValueError("Tracking decoded-frame count does not match timestamps")
    gallery = output / "visualization" / f"{state}_temporal.html"
    gallery.parent.mkdir(parents=True, exist_ok=True)
    gallery.write_text(
        '<!doctype html><meta charset="utf-8"><title>Temporal observations</title><style>body{font-family:system-ui;background:#101820;color:white;padding:20px}main{display:grid;grid-template-columns:repeat(auto-fit,minmax(220px,1fr));gap:12px}img{width:100%;height:160px;object-fit:contain}article{background:#203040;padding:12px}p{overflow-wrap:anywhere}</style><h1>Neural temporal observations</h1><p>Proposed labels and uncalibrated 2D tracks. These are not verified changes.</p><main>'
        + "".join(cards)
        + "</main>"
    )
    return {
        "decoded_frames": index,
        "runtime_seconds": time.perf_counter() - started,
        "diagnostics": str(destination),
        "observation_gallery": str(gallery),
        "neural_inference_every_frame": False,
        "neural_redetection_calls": redetection_count,
        "supported_temporal_identity_links": identity_links,
        "method": "Neural re-detection plus forward/backward flow; mutual unambiguous 2D identity linking",
        "cross_anchor_identity_verified": False,
    }
