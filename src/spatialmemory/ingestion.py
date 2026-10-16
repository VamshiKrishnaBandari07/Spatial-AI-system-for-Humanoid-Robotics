"""Read-only video ingestion; bounded selection with access to all original frames."""

import hashlib, json, re, subprocess
from pathlib import Path
import cv2
import numpy as np
from .contracts import Config, Frame


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1024 * 1024), b""):
            h.update(b)
    return h.hexdigest()


def _ffprobe(path: Path, *arguments: str) -> dict:
    """Run ffprobe and return its JSON. Failures name the file, never server paths."""
    result = subprocess.run(
        ["ffprobe", "-v", "error", *arguments, "-of", "json", str(path)],
        capture_output=True,
        text=True,
    )
    if result.returncode:
        lines = (result.stderr or "").replace(str(path), path.name).strip().splitlines()
        # Drop ffprobe's "[demuxer @ 0xADDRESS]" prefix: it is internal noise.
        reason = re.sub(r"^\[[^\]]*\]\s*", "", lines[0]) if lines else ""
        raise ValueError(
            f"Cannot read {path.name} as a video file" + (f": {reason}" if reason else "")
        )
    try:
        return json.loads(result.stdout)
    except json.JSONDecodeError as error:
        raise ValueError(f"Unreadable video metadata in {path.name}") from error


def ingest(path: Path, output: Path, config: Config, prefix: str, timestamps=None):
    path = path.resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    metadata = _ffprobe(path, "-show_streams", "-show_format")
    timing = _ffprobe(
        path,
        "-select_streams",
        "v:0",
        "-show_frames",
        "-show_entries",
        "frame=best_effort_timestamp_time",
    )
    original_times = [
        float(f["best_effort_timestamp_time"])
        for f in timing.get("frames", [])
        if "best_effort_timestamp_time" in f
    ]
    if timestamps is not None:
        original_times = list(timestamps)
    if (
        not original_times
        or not np.isfinite(original_times).all()
        or np.any(np.diff(original_times) <= 0)
    ):
        raise ValueError("Missing or non-increasing original video timestamps")
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise ValueError(f"Cannot decode {path.name}")
    orientation_enabled = cap.set(cv2.CAP_PROP_ORIENTATION_AUTO, 1)
    rotations = [
        s.get("rotation", 0)
        for stream in metadata["streams"]
        for s in stream.get("side_data_list", [])
    ]
    if any(rotations) and not orientation_enabled:
        cap.release()
        raise ValueError("Decoder cannot apply video orientation metadata")
    output.mkdir(parents=True, exist_ok=True)
    fps = cap.get(cv2.CAP_PROP_FPS)
    if not np.isfinite(fps) or fps <= 0:
        raise ValueError("Video has invalid frame rate")
    frames = []
    index = 0
    previous = None
    next_sample = original_times[0]
    capped = (
        original_times[-1] - original_times[0]
    ) / config.sample_seconds > config.max_frames - 1
    selected_indices = None
    if capped:
        targets = np.linspace(original_times[0], original_times[-1], config.max_frames)
        selected_indices = set(
            np.minimum(
                np.searchsorted(original_times, targets), len(original_times) - 1
            ).tolist()
        )
    orb = cv2.ORB_create(nfeatures=1500)
    try:
        while True:
            ok, image = cap.read()
            if not ok:
                break
            if index >= len(original_times):
                raise ValueError("Decoded frame count exceeds timestamp count")
            timestamp = original_times[index]
            if (
                (index in selected_indices) if capped else timestamp >= next_sample
            ) and len(frames) < config.max_frames:
                next_sample = timestamp + config.sample_seconds
                if image.shape[1] > config.max_width:
                    image = cv2.resize(
                        image,
                        (
                            config.max_width,
                            round(image.shape[0] * config.max_width / image.shape[1]),
                        ),
                    )
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                small = cv2.resize(gray, (32, 32)).astype(float) / 255
                distance = (
                    None if previous is None else float(np.mean(abs(small - previous)))
                )
                file = output / f"{prefix}_{index:06d}.jpg"
                if not cv2.imwrite(str(file), image):
                    raise OSError(f"Failed to write {file}")
                frames.append(
                    Frame(
                        id=f"{prefix}:{index}",
                        index=index,
                        timestamp=timestamp,
                        path=str(file),
                        sharpness=float(cv2.Laplacian(gray, cv2.CV_64F).var()),
                        exposure=float(gray.mean() / 255),
                        features=len(orb.detect(gray, None)),
                        duplicate_distance=distance,
                    )
                )
                previous = small
            index += 1
    finally:
        cap.release()
    if not frames:
        raise ValueError("Video contains no decodable frames")
    if index != len(original_times):
        raise ValueError("Decoded frames and timestamp records disagree")
    result = {
        "source": str(path),
        "sha256": sha256(path),
        "decoded_frames": index,
        "original_timestamps_seconds": original_times,
        "fps": fps,
        "metadata": metadata,
        "orientation": "OpenCV auto orientation enabled",
        "selection_limit_reached": capped,
        "selection_policy": "bounded timestamps distributed over complete recording"
        if capped
        else "timestamp interval sampling",
        "coverage_timestamp_range": [frames[0].timestamp, frames[-1].timestamp],
        "frames": [f.model_dump() for f in frames],
    }
    (output / "manifest.json").write_text(json.dumps(result, indent=2))
    return frames, result


def keyframes(frames, limit=24):
    candidates = [
        f
        for f in frames
        if f.features >= 40
        and 0.05 < f.exposure < 0.95
        and (f.duplicate_distance is None or f.duplicate_distance > 0.005)
    ]
    if len(candidates) < 2:
        raise ValueError("Insufficient distinct feature-rich frames for reconstruction")
    if len(candidates) <= limit:
        return candidates
    # Temporal spread is a proxy only; not independent viewpoint confirmation.
    ids = np.linspace(0, len(candidates) - 1, limit).round().astype(int)
    return [candidates[i] for i in ids]


def neural_keyframes(frames, count):
    """Feature-quality selection with temporal spread; not a viewpoint guarantee."""
    candidates = keyframes(frames)
    maximum = max(f.features for f in candidates)
    candidates = [f for f in candidates if f.features >= maximum * 0.1]
    if len(candidates) < 2:
        raise ValueError("Insufficient quality-qualified neural frames")
    selected = [max(candidates, key=lambda f: (f.features, f.sharpness))]
    span = max(f.timestamp for f in candidates) - min(f.timestamp for f in candidates)
    while len(selected) < min(count, len(candidates)):
        remaining = [f for f in candidates if f not in selected]
        selected.append(
            max(
                remaining,
                key=lambda f: (f.features / maximum) ** 0.5
                * min(abs(f.timestamp - s.timestamp) for s in selected)
                / max(span, 1e-9),
            )
        )
    return sorted(selected, key=lambda f: f.index)
