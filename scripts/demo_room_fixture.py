"""Synthetic indoor-like pair with a removable box object for UI demos.

Not a smartphone accuracy benchmark. Creates H.264 MP4s with wall panels plus a
bright box that is present in BEFORE and absent in AFTER so object analysis has
something to detect. Do not treat results as real-world change-detection proof.
"""

from pathlib import Path
import argparse
import subprocess
import cv2
import numpy as np


def _panel(rng, size=96):
    texture = rng.integers(40, 180, (size, size, 3), dtype=np.uint8)
    texture = cv2.GaussianBlur(texture, (5, 5), 0)
    for _ in range(20):
        a, b = rng.integers(4, size - 4, 2)
        cv2.circle(
            texture,
            (int(a), int(b)),
            int(rng.integers(2, 8)),
            tuple(int(v) for v in rng.integers(0, 255, 3)),
            -1,
        )
    return texture


def _draw_box(image, center, size, color):
    x, y = center
    half = size // 2
    cv2.rectangle(
        image,
        (x - half, y - half),
        (x + half, y + half),
        color,
        thickness=-1,
    )
    cv2.rectangle(
        image,
        (x - half, y - half),
        (x + half, y + half),
        (20, 20, 20),
        thickness=3,
    )
    # Lid highlight so it reads as a box/cabinet-like object.
    cv2.rectangle(
        image,
        (x - half + 6, y - half + 6),
        (x + half - 6, y - half + max(12, size // 5)),
        tuple(min(255, c + 40) for c in color),
        thickness=-1,
    )


def make_frame(i, rng, panels, include_object: bool):
    w, h, focal = 640, 480, 600
    image = np.full((h, w, 3), 28, dtype=np.uint8)
    cam = np.array([i * 0.05, 0.0, 0.0])
    for center, texture in sorted(panels, key=lambda p: -p[0][2]):
        p = center - cam
        size = int(focal * 0.5 / p[2])
        u = int(focal * p[0] / p[2] + w / 2)
        v = int(focal * p[1] / p[2] + h / 2)
        x, y = u - size // 2, v - size // 2
        if x >= 0 and y >= 0 and x + size < w and y + size < h and size > 8:
            image[y : y + size, x : x + size] = cv2.resize(texture, (size, size))
    # Static furniture-like cabinet on the left.
    _draw_box(image, (140, 300), 110, (90, 70, 55))
    # Table-like slab.
    cv2.rectangle(image, (250, 340), (520, 390), (120, 110, 95), -1)
    cv2.rectangle(image, (250, 340), (520, 390), (30, 30, 30), 2)
    if include_object:
        # Bright pink handled container — present only in BEFORE.
        _draw_box(image, (380 + i, 250), 70, (40, 60, 220))
        cv2.rectangle(image, (360 + i, 235), (400 + i, 248), (30, 40, 180), -1)
    return image


def write_raw(path: Path, include_object: bool, seed: int, frames: int = 18):
    rng = np.random.default_rng(seed)
    panels = []
    for y in range(-2, 3):
        for x in range(-3, 4):
            z = float(rng.uniform(4.0, 7.0))
            panels.append((np.array([x * 0.55, y * 0.45, z]), _panel(rng)))
    tmp = path.with_suffix(".raw.mp4")
    writer = cv2.VideoWriter(str(tmp), cv2.VideoWriter_fourcc(*"mp4v"), 6, (640, 480))
    if not writer.isOpened():
        raise RuntimeError("Video encoder unavailable")
    for i in range(frames):
        writer.write(make_frame(i, rng, panels, include_object))
    writer.release()
    subprocess.check_call(
        [
            "ffmpeg",
            "-y",
            "-i",
            str(tmp),
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-movflags",
            "+faststart",
            str(path),
        ],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    tmp.unlink(missing_ok=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    args = parser.parse_args()
    args.directory.mkdir(parents=True, exist_ok=True)
    write_raw(args.directory / "before.mp4", include_object=True, seed=7)
    write_raw(args.directory / "after.mp4", include_object=False, seed=7)
    print({"before": str(args.directory / "before.mp4"), "after": str(args.directory / "after.mp4")})


if __name__ == "__main__":
    main()
