"""Segmented cross-video visual correspondence; never physical change proof."""

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from .geometry import match


def frames(video, interval, segment_seconds):
    if interval < 0 or segment_seconds <= 0:
        raise ValueError("Interval must be nonnegative and segment duration positive")
    cap = cv2.VideoCapture(str(video))
    sift = cv2.SIFT_create(nfeatures=800)
    result, last, index = [], -float("inf"), 0
    try:
        if not cap.isOpened():
            raise ValueError(f"Cannot decode {video}")
        while True:
            ok, image = cap.read()
            if not ok:
                break
            timestamp = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000
            if timestamp - last >= interval:
                h, w = image.shape[:2]
                image = cv2.resize(image, (480, round(h * 480 / w)))
                kp, descriptor = sift.detectAndCompute(
                    cv2.cvtColor(image, cv2.COLOR_BGR2GRAY), None
                )
                result.append(
                    dict(
                        index=index,
                        timestamp=timestamp,
                        segment=int(timestamp // segment_seconds),
                        image=image,
                        points=np.asarray([k.pt for k in kp]),
                        descriptor=descriptor,
                    )
                )
                last = timestamp
            index += 1
    finally:
        cap.release()
    if not result:
        raise ValueError(f"No valid frames in {video}")
    return result


def correspondence(a, b):
    pairs = match(a["descriptor"], b["descriptor"])
    if len(pairs) < 12:
        return None
    ia, ib = np.asarray(pairs).T
    matrix, mask = cv2.findHomography(a["points"][ia], b["points"][ib], cv2.RANSAC, 3)
    if matrix is None or mask is None or not np.isfinite(matrix).all():
        return None
    inliers = int(mask.sum())
    ratio = inliers / len(pairs)
    if inliers < 12 or ratio < 0.5:
        return None
    return {
        "inliers": inliers,
        "inlier_ratio": ratio,
        "homography_before_to_after": matrix.tolist(),
    }


def compare(before, after, output, interval=0.5, segment_seconds=5):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    a, b = (
        frames(before, interval, segment_seconds),
        frames(after, interval, segment_seconds),
    )
    rows = []
    for frame in a:
        candidates = []
        for other in b:
            evidence = correspondence(frame, other)
            if evidence:
                candidates.append(
                    (evidence["inliers"] * evidence["inlier_ratio"], other, evidence)
                )
        candidates.sort(key=lambda item: item[0], reverse=True)
        row = {
            "before_frame": frame["index"],
            "before_timestamp": frame["timestamp"],
            "before_segment": frame["segment"],
            "physical_change_verified": False,
        }
        if not candidates or (
            len(candidates) > 1 and candidates[0][0] < 1.05 * candidates[1][0]
        ):
            row["status"] = "no_unique_supported_view"
        else:
            _, other, evidence = candidates[0]
            image = np.concatenate(
                (
                    frame["image"],
                    cv2.resize(
                        other["image"],
                        (frame["image"].shape[1], frame["image"].shape[0]),
                    ),
                ),
                axis=1,
            )
            name = f"pair_{frame['index']}_{other['index']}.jpg"
            if not cv2.imwrite(str(output / name), image):
                raise ValueError("Cannot write comparison evidence")
            row.update(
                status="visual_correspondence",
                after_frame=other["index"],
                after_timestamp=other["timestamp"],
                after_segment=other["segment"],
                evidence_image=name,
                geometry=evidence,
            )
        rows.append(row)
    report = {
        "scope": "2D viewpoint comparison, not removal detection; homography may fail with parallax",
        "interval_seconds": interval,
        "segment_seconds": segment_seconds,
        "before_frames_compared": len(a),
        "after_candidates": len(b),
        "comparisons": rows,
    }
    (output / "comparison.json").write_text(json.dumps(report, indent=2))
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("before", type=Path)
    parser.add_argument("after", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument(
        "--interval",
        type=float,
        default=0.5,
        help="0 compares every decoded frame (expensive)",
    )
    parser.add_argument("--segment-seconds", type=float, default=5)
    args = parser.parse_args()
    report = compare(
        args.before, args.after, args.output, args.interval, args.segment_seconds
    )
    print(json.dumps({k: v for k, v in report.items() if k != "comparisons"}))


if __name__ == "__main__":
    main()
