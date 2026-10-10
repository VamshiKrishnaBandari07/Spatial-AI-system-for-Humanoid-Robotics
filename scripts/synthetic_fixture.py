"""Explicitly synthetic nonplanar textured scene, NOT a smartphone benchmark."""

from pathlib import Path
import cv2, numpy as np


def make_video(path, seed=4, frames=18):
    rng = np.random.default_rng(seed)
    w, h = 640, 480
    focal = 600
    panels = []
    for y in range(-2, 3):
        for x in range(-3, 4):
            z = float(rng.uniform(4, 7))
            center = np.array([x * 0.55, y * 0.5, z])
            texture = rng.integers(0, 256, (64, 64, 3), dtype=np.uint8)
            texture = cv2.GaussianBlur(texture, (3, 3), 0)
            for _ in range(12):
                a, b = rng.integers(3, 60, 2)
                cv2.circle(
                    texture,
                    (int(a), int(b)),
                    int(rng.integers(2, 7)),
                    tuple(int(v) for v in rng.integers(0, 256, 3)),
                    -1,
                )
            panels.append((center, texture))
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 6, (w, h))
    if not writer.isOpened():
        raise RuntimeError("Video encoder unavailable")
    for i in range(frames):
        image = np.full((h, w, 3), 25, dtype=np.uint8)
        cam = np.array([i * 0.065, 0, 0])
        for center, texture in sorted(panels, key=lambda p: -p[0][2]):
            p = center - cam
            size = int(focal * 0.43 / p[2])
            u = int(focal * p[0] / p[2] + w / 2)
            v = int(focal * p[1] / p[2] + h / 2)
            x, y = u - size // 2, v - size // 2
            if x >= 0 and y >= 0 and x + size < w and y + size < h:
                image[y : y + size, x : x + size] = cv2.resize(texture, (size, size))
        writer.write(image)
    writer.release()
    return path


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument("directory", type=Path)
    a = p.parse_args()
    a.directory.mkdir(parents=True, exist_ok=True)
    make_video(a.directory / "before.mp4")
    make_video(a.directory / "after.mp4")
