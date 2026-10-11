"""Every-frame 2D propagation between real neural anchors, never 3D change proof."""

import json
from pathlib import Path
import cv2
from .segmentation import propagate_mask


def track_recording(video, output, state, width=320):
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
                if previous is not None:
                    for track_id, mask in list(active.items()):
                        propagated, audit = propagate_mask(previous, image, mask)
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
                            del active[track_id]
                        else:
                            active[track_id] = propagated
                if index in anchors:
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
    return {
        "decoded_frames": index,
        "diagnostics": str(destination),
        "neural_inference_every_frame": False,
        "method": "Neural anchor masks plus forward/backward optical flow; drift stops propagation",
        "cross_anchor_identity_verified": False,
    }
