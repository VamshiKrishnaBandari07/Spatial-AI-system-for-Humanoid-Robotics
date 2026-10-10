"""Dataset conversion uses documented real rescan relationships or supplied pair evidence.
No dataset is downloaded; users must obtain data under its own terms.
"""

import json, struct
from pathlib import Path
import cv2, numpy as np
from scipy.spatial.transform import Rotation
from .inputs import Manifest, Recording, Camera, Poses, Pose

SOURCES = {
    "3rscan": {
        "source": "https://github.com/WaldJohannaU/3RScan",
        "license": "Dataset terms must be accepted at source; not bundled",
    },
    "scannet": {
        "source": "https://github.com/ScanNet/ScanNet",
        "license": "ScanNet terms of use; obtain authorized data separately",
    },
}


def encode_images(images, path, fps=30):
    if not images:
        raise ValueError("No RGB frames found")
    first = cv2.imread(str(images[0]))
    if first is None:
        raise ValueError("Cannot decode first RGB frame")
    h, w = first.shape[:2]
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    if not writer.isOpened():
        raise RuntimeError("MP4 encoder unavailable")
    try:
        for file in images:
            image = cv2.imread(str(file))
            if image is None or image.shape[:2] != (h, w):
                raise ValueError("Invalid or inconsistent RGB sequence")
            writer.write(image)
    finally:
        writer.release()
    return w, h


def info_file(path):
    result = {}
    for line in path.read_text().splitlines():
        if "=" in line:
            k, v = line.split("=", 1)
            result[k.strip()] = v.strip()
    return result


def export_sequence(source, dest, dataset):
    seq = source / "sequence" if (source / "sequence").is_dir() else source
    images = sorted(seq.glob("frame-*.color.jpg")) or sorted(
        (source / "color").glob("*.jpg")
    )
    dest.mkdir(parents=True, exist_ok=False)
    w, h = encode_images(images, dest / "rgb.mp4")
    times = [i / 30 for i in range(len(images))]
    (dest / "timestamps.json").write_text(json.dumps(times))
    camera_path = None
    pose_path = None
    depth_path = None
    unavailable = []
    sensor = None
    if (source / "sensor_metadata.json").exists():
        sensor = json.loads((source / "sensor_metadata.json").read_text())
        times = [t["rgb_timestamp_us"] / 1e6 for t in sensor["timestamps"]]
        if len(times) != len(images):
            raise ValueError("Native timestamp/frame count mismatch")
        (dest / "timestamps.json").write_text(json.dumps(times))
    if (seq / "_info.txt").exists():
        info = info_file(seq / "_info.txt")
        raw = info.get("m_calibrationColorIntrinsic")
        if raw:
            matrix = np.array([float(x) for x in raw.split()]).reshape(4, 4)
            camera = Camera(
                width=int(info.get("m_colorWidth", w)),
                height=int(info.get("m_colorHeight", h)),
                fx=matrix[0, 0],
                fy=matrix[1, 1],
                cx=matrix[0, 2],
                cy=matrix[1, 2],
                source=f"{dataset} native calibration",
            )
            camera_path = dest / "camera.json"
            camera_path.write_text(camera.model_dump_json(indent=2))
    elif (source / "intrinsic/intrinsic_color.txt").exists():
        matrix = np.loadtxt(source / "intrinsic/intrinsic_color.txt")
        camera = Camera(
            width=w,
            height=h,
            fx=matrix[0, 0],
            fy=matrix[1, 1],
            cx=matrix[0, 2],
            cy=matrix[1, 2],
            source="ScanNet extracted calibration",
        )
        camera_path = dest / "camera.json"
        camera_path.write_text(camera.model_dump_json(indent=2))
    poses = []
    for i, image in enumerate(images):
        file = (
            image.with_name(image.name.replace(".color.jpg", ".pose.txt"))
            if dataset == "3rscan"
            else source / "pose" / (image.stem + ".txt")
        )
        if file.exists():
            matrix = np.loadtxt(file)
            if matrix.shape == (4, 4) and np.isfinite(matrix).all():
                poses.append(
                    Pose(
                        timestamp_ns=round(times[i] * 1e9),
                        translation=matrix[:3, 3].tolist(),
                        quaternion_xyzw=Rotation.from_matrix(matrix[:3, :3])
                        .as_quat()
                        .tolist(),
                    )
                )
    if poses:
        pose_path = dest / "poses.json"
        pose_path.write_text(
            Poses(
                translation_unit="metres",
                source=f"{dataset} native camera-to-world",
                poses=poses,
            ).model_dump_json(indent=2)
        )
    if not camera_path:
        unavailable.append("calibration")
    if len(poses) != len(images):
        unavailable.append("some or all camera poses")
    if dataset == "3rscan" and (seq / "_info.txt").exists():
        from .inputs import DepthMetadata, DepthFrame

        info = info_file(seq / "_info.txt")
        raw = info.get("m_calibrationDepthIntrinsic")
        shift = float(info.get("m_depthShift", "0"))
        depth_files = [
            image.with_name(image.name.replace(".color.jpg", ".depth.pgm"))
            for image in images
        ]
        if raw and shift > 0 and all(p.exists() for p in depth_files):
            depth_path = dest / "depth"
            depth_path.mkdir()
            K = np.array([float(v) for v in raw.split()]).reshape(4, 4)
            first_depth = cv2.imread(str(depth_files[0]), cv2.IMREAD_UNCHANGED)
            dh, dw = first_depth.shape
            dc = Camera(
                width=dw,
                height=dh,
                fx=K[0, 0],
                fy=K[1, 1],
                cx=K[0, 2],
                cy=K[1, 2],
                source="3RScan native depth calibration",
            )
            dcam = dest / "depth_camera.json"
            dcam.write_text(dc.model_dump_json(indent=2))
            entries = []
            for i, file in enumerate(depth_files):
                depth = cv2.imread(str(file), cv2.IMREAD_UNCHANGED)
                if depth is None or depth.shape != (dh, dw):
                    raise ValueError("Invalid native 3RScan depth frame")
                filename = f"{i:06d}.npy"
                np.save(depth_path / filename, depth.astype(float) / shift)
                entries.append(
                    DepthFrame(
                        file=filename,
                        timestamp_ns=round(times[i] * 1e9),
                        rgb_frame_index=i,
                    )
                )
            # Native RGB/depth extrinsics are required; do not silently assume alignment.
            ce = info.get("m_calibrationColorExtrinsic")
            de = info.get("m_calibrationDepthExtrinsic")
            if ce and de:
                Ec = np.array([float(v) for v in ce.split()]).reshape(4, 4)
                Ed = np.array([float(v) for v in de.split()]).reshape(4, 4)
                dm = DepthMetadata(
                    unit="metres",
                    width=dw,
                    height=dh,
                    camera=str(dcam),
                    rgb_aligned=False,
                    T_rgb_depth=(np.linalg.inv(Ec) @ Ed).tolist(),
                    scale_verified=True,
                    frames=entries,
                )
                (depth_path / "metadata.json").write_text(dm.model_dump_json(indent=2))
            else:
                unavailable.append(
                    "3RScan depth extrinsics: raw depth retained but not usable for fusion"
                )
                depth_path = None
    if dataset == "scannet" and sensor is not None:
        from .inputs import DepthMetadata, DepthFrame
        import shutil

        depth_path = dest / "depth"
        depth_path.mkdir()
        dw, dh = sensor["depth_dimensions"]
        K = np.loadtxt(source / "intrinsic/intrinsic_depth.txt")
        dc = Camera(
            width=dw,
            height=dh,
            fx=K[0, 0],
            fy=K[1, 1],
            cx=K[0, 2],
            cy=K[1, 2],
            source="ScanNet native depth calibration",
        )
        dcam = dest / "depth_camera.json"
        dcam.write_text(dc.model_dump_json(indent=2))
        entries = []
        for i, t in enumerate(sensor["timestamps"]):
            filename = f"{i:06d}.npy"
            shutil.copy2(source / "depth" / filename, depth_path / filename)
            entries.append(
                DepthFrame(
                    file=filename,
                    timestamp_ns=t["depth_timestamp_us"] * 1000,
                    rgb_frame_index=i,
                )
            )
        Ec = np.loadtxt(source / "intrinsic/extrinsic_color.txt")
        Ed = np.loadtxt(source / "intrinsic/extrinsic_depth.txt")
        dm = DepthMetadata(
            unit="metres",
            width=dw,
            height=dh,
            camera=str(dcam),
            rgb_aligned=False,
            T_rgb_depth=(np.linalg.inv(Ec) @ Ed).tolist(),
            scale_verified=True,
            frames=entries,
        )
        (depth_path / "metadata.json").write_text(dm.model_dump_json(indent=2))
    unavailable += ["IMU", "change annotations"]
    if depth_path is None:
        unavailable.append("depth conversion")
    return Recording(
        video=str(dest / "rgb.mp4"),
        camera=str(camera_path) if camera_path else None,
        poses=str(pose_path) if pose_path else None,
        depth_dir=str(depth_path) if depth_path else None,
        timestamps=str(dest / "timestamps.json"),
    ), unavailable


def convert_3rscan(root, pairs_json, reference, rescan, output, split):
    relations = json.loads(Path(pairs_json).read_text())
    valid = any(
        entry.get("reference") == reference
        and any(s.get("reference") == rescan for s in entry.get("scans", []))
        for entry in relations
    )
    if not valid:
        raise ValueError(
            "Pair is not a documented 3RScan reference/rescan relationship"
        )
    return convert_pair(
        Path(root) / reference,
        Path(root) / rescan,
        output,
        "3rscan",
        f"{reference}__{rescan}",
        split,
        "3RScan native rescan relation",
    )


def convert_scannet(root, before_scan, after_scan, pair_evidence, output, split):
    if before_scan == after_scan:
        raise ValueError("Same ScanNet recording cannot be labeled a changed pair")
    evidence = json.loads(Path(pair_evidence).read_text())
    if (
        evidence.get("before") != before_scan
        or evidence.get("after") != after_scan
        or not evidence.get("same_environment_verified")
        or not evidence.get("source")
    ):
        raise ValueError(
            "ScanNet needs explicit verified same-environment pair metadata; unrelated scans are rejected"
        )
    return convert_pair(
        Path(root) / before_scan,
        Path(root) / after_scan,
        output,
        "scannet",
        f"{before_scan}__{after_scan}",
        split,
        evidence["source"],
    )


def convert_pair(before, after, output, dataset, scene, split, pair_source):
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    b, bu = export_sequence(before, output / "before", dataset)
    a, au = export_sequence(after, output / "after", dataset)
    manifest = Manifest(
        scene_id=scene,
        before=b,
        after=a,
        dataset={
            "name": dataset,
            **SOURCES[dataset],
            "split": split,
            "pair_source": pair_source,
            "source_directories": [str(before), str(after)],
            "unavailable": {"before": bu, "after": au},
            "timestamps": "Native timestamp sidecars when available; otherwise documented 30 fps synthetic encoding schedule",
        },
    )
    path = output / "manifest.json"
    path.write_text(manifest.model_dump_json(indent=2))
    return path


def extract_scannet_sens(path, output):
    """ScanNet SensorData version 4: JPEG/PNG RGB and zlib uint16 depth only.
    Supported binary format validated against ScanNet's official SensorData reader.
    """
    import zlib

    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    for folder in ["color", "depth", "pose", "intrinsic"]:
        (output / folder).mkdir()

    def read(f, fmt):
        size = struct.calcsize("<" + fmt)
        data = f.read(size)
        if len(data) != size:
            raise ValueError("Truncated .sens file")
        return struct.unpack("<" + fmt, data)

    with open(path, "rb") as f:
        if read(f, "I")[0] != 4:
            raise ValueError("Only .sens version 4 supported")
        length = read(f, "Q")[0]
        if length > 1024 * 1024:
            raise ValueError("Invalid sensor name length")
        name = f.read(length).decode("utf-8")
        matrices = [np.array(read(f, "16f")).reshape(4, 4) for _ in range(4)]
        cc, dc = read(f, "ii")
        cw, ch, dw, dh = read(f, "IIII")
        shift = read(f, "f")[0]
        count = read(f, "Q")[0]
        if cc not in [1, 2] or dc != 1:
            raise ValueError("Requires PNG/JPEG color and zlib ushort depth")
        if max(cw, ch, dw, dh) > 16384 or count > 1000000 or shift <= 0:
            raise ValueError("Invalid sensor dimensions/frame count/depth shift")
        timestamps = []
        for i in range(count):
            pose = np.array(read(f, "16f")).reshape(4, 4)
            tc, td, cs, ds = read(f, "QQQQ")
            if cs > 100000000 or ds > 100000000:
                raise ValueError("Invalid compressed frame size")
            color = f.read(cs)
            depth = f.read(ds)
            if len(color) != cs or len(depth) != ds:
                raise ValueError("Truncated compressed frame")
            image = cv2.imdecode(np.frombuffer(color, np.uint8), cv2.IMREAD_COLOR)
            if image is None or image.shape[:2] != (ch, cw):
                raise ValueError("Bad .sens color image")
            decoder = zlib.decompressobj()
            raw = decoder.decompress(depth, dw * dh * 2 + 1)
            if len(raw) != dw * dh * 2 or not decoder.eof:
                raise ValueError("Bad .sens depth payload")
            dep = np.frombuffer(raw, dtype="<u2").reshape(dh, dw).astype(float) / shift
            cv2.imwrite(str(output / "color" / f"{i:06d}.jpg"), image)
            np.save(output / "depth" / f"{i:06d}.npy", dep)
            np.savetxt(output / "pose" / f"{i:06d}.txt", pose)
            timestamps.append({"rgb_timestamp_us": tc, "depth_timestamp_us": td})
        for key, matrix in zip(
            [
                "intrinsic_color",
                "extrinsic_color",
                "intrinsic_depth",
                "extrinsic_depth",
            ],
            matrices,
        ):
            np.savetxt(output / "intrinsic" / f"{key}.txt", matrix)
    (output / "sensor_metadata.json").write_text(
        json.dumps(
            {
                "sensor": name,
                "source": str(path),
                "depth_unit": "metres",
                "timestamps": timestamps,
                "color_dimensions": [cw, ch],
                "depth_dimensions": [dw, dh],
            },
            indent=2,
        )
    )
    return output
