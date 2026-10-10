"""Read-only native 3RScan input checks; no labels, downloads or model changes."""

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from spatialmemory.datasets import info_file
from spatialmemory.inputs import load_manifest
from spatialmemory.rgbd_validation import project, transform_points, unproject


def check_scan(root):
    sequence = root / "extracted"
    info = info_file(sequence / "_info.txt")
    K = np.array(
        [float(x) for x in info["m_calibrationDepthIntrinsic"].split()]
    ).reshape(4, 4)[:3, :3]
    shift = float(info["m_depthShift"])
    if not np.isfinite(shift) or shift <= 0:
        raise ValueError("Invalid native depth scale")
    images = sorted(sequence.glob("*.color.jpg"))
    if len(images) != int(info["m_frames.size"]) or not images:
        raise ValueError("Native frame count mismatch")
    for path in images:
        stem = path.name.removesuffix(".color.jpg")
        rgb = cv2.imread(str(path))
        depth = cv2.imread(str(sequence / (stem + ".depth.pgm")), cv2.IMREAD_UNCHANGED)
        if rgb is None or rgb.shape[:2] != (
            int(info["m_colorHeight"]),
            int(info["m_colorWidth"]),
        ):
            raise ValueError("Invalid native RGB frame")
        if depth is None or depth.shape != (
            int(info["m_depthHeight"]),
            int(info["m_depthWidth"]),
        ):
            raise ValueError("Invalid native depth frame")
        metric = depth.astype(float) / shift
        metric[depth == 0] = np.nan
        points, pixels = unproject(metric, K, stride=8)
        pose = np.loadtxt(sequence / (stem + ".pose.txt"))
        world = transform_points(points, pose)
        if (
            not len(points)
            or not np.allclose(project(points, K), pixels, atol=1e-8)
            or not np.allclose(
                transform_points(world, np.linalg.inv(pose)), points, atol=1e-8
            )
        ):
            raise ValueError("Depth/pose coordinate round trip failed")
    annotations = json.loads((root / "semseg.v2.json").read_text())
    ids = [int(a["objectId"]) for a in annotations["segGroups"]]
    if annotations["scan_id"] != root.name or len(ids) != len(set(ids)):
        raise ValueError("Native semantic scan or identity mismatch")
    with (root / "labels.instances.annotated.v2.ply").open("rb") as stream:
        header = []
        while True:
            line = stream.readline()
            if not line or len(header) > 1000:
                raise ValueError("Truncated or oversized PLY header")
            header.append(line.decode().strip())
            if header[-1] == "end_header":
                break
        if "format ascii 1.0" not in header:
            raise ValueError("This validator supports the acquired ASCII PLY subset")
        element = next(h for h in header if h.startswith("element vertex "))
        count = int(element.split()[-1])
        properties = []
        for line in header[header.index(element) + 1 :]:
            if line.startswith("element "):
                break
            if line.startswith("property "):
                properties.append(line.split()[-1])
        vertices = np.loadtxt(stream, max_rows=count, ndmin=2)
    if len(vertices) != count or not np.isfinite(vertices).all():
        raise ValueError("Invalid annotated PLY vertices")
    object_ids = vertices[:, properties.index("objectId")]
    if not np.equal(object_ids, np.floor(object_ids)).all() or set(
        object_ids.astype(int)
    ) - set(ids) - {0}:
        raise ValueError("PLY object IDs disagree with semantic annotations")
    return {
        "scan_id": root.name,
        "frames_checked": len(images),
        "annotated_vertices_checked": count,
        "semantic_objects": len(ids),
        "status": "passed",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    args = parser.parse_args()
    plan = json.loads(args.plan.read_text())
    load_manifest(args.manifest, Path.cwd())
    root = Path(plan["raw_destination"])
    print(
        json.dumps(
            {
                "scans": [
                    check_scan(root / plan[key]) for key in ["reference", "rescan"]
                ]
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
