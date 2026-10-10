# Canonical input v1.0

The core command accepts two RGB videos directly or a manifest conforming to `Manifest` in `src/spatialmemory/inputs.py`. The scene ID is required in manifests. Direct commands default to `direct_input` and accept `--scene-id`. Supported containers: MP4, MOV, MKV. Codec support depends on the installed FFmpeg/OpenCV decoder; H.264/H.265 are expected but have not been tested with supplied smartphone recordings.

Paths in a manifest resolve against `--project-root`, defaulting to the current directory. Absolute and symlink-resolved paths must stay within this root. A dataset conversion written outside the repository must use a project root covering its output. Missing before/after videos fail; absent optional files produce warnings. A malformed present optional file fails rather than silently discarding data. Originals are never edited.

`examples/scene_manifest.json` uses the requested `data/raw/scene_001/{before,after}/rgb.mp4` structure. Annotations are optional during inference. `examples/annotations_template.json` is empty by design: illustrative object coordinates would not constitute ground truth.

## Optional metadata

Camera JSON specifies PINHOLE width, height, fx, fy, cx, cy, distortion_model (`none` or `opencv`), coefficients and provenance `source`. Example numbers are not measured calibration. Provide intrinsics in the decoded display orientation. Dimensions rescale to processing resolution.

Pose JSON uses `T_world_camera`, translation_unit, source/destination frames and a `poses` array. Each entry contains timestamp_ns, translation and either quaternion_xyzw or a proper rotation_matrix. TXT/NPY files use N×8 rows `timestamp_ns tx ty tz qx qy qz qw` and an adjacent `.metadata.json` containing the convention, units and provenance. Arrays must be numerical and are loaded without pickled objects. Quaternion normalization and timestamp ordering are validated. Pose and video timelines must overlap.

Depth directories need `metadata.json`: schema_version, unit (`metres`, `millimetres`, `relative`), width/height, invalid_value, camera path, rgb_aligned, optional T_rgb_depth, scale_verified and frames. Each frame declares file, timestamp_ns and rgb_frame_index. PNG/TIFF/NPY and NPZ with a `depth` key are supported. Depth dimensions and calibration are validated. Unaligned depth requires an explicit 4×4 transform; actual reprojection/fusion is not yet integrated. RGB references and depth timestamp synchronization are checked within a documented 50 ms tolerance. Relative depth is never silently treated as metric.

IMU CSV columns are `timestamp_ns,ax,ay,az,gx,gy,gz`. Adjacent `.metadata.json` must specify `acceleration_unit: "m/s^2"`, `angular_velocity_unit: "rad/s"`, time_offset_ns and calibration provenance. Ordered finite samples are checked, and shifted timestamps must overlap RGB. IMU is retained and validated, not yet fused for pose estimation.

## Dataset conversion

Obtain datasets directly under their own terms. No dataset or annotations are fabricated or downloaded automatically.

```bash
python scripts/convert_dataset.py sens scan.sens /workspace/datasets/extracted_scan
python scripts/convert_dataset.py 3rscan --root /workspace/datasets/3rscan --before REFERENCE_UUID --after RESCAN_UUID --pair-metadata /workspace/datasets/3RScan.json --split validation --output /workspace/datasets/canonical_pair
python scripts/convert_dataset.py scannet --root /workspace/datasets/scannet --before scene0000_00 --after scene0000_01 --pair-metadata /workspace/datasets/pair.json --split validation --output /workspace/datasets/canonical_scannet
```

3RScan pairs must occur in the native reference/rescan relationship JSON. ScanNet pair metadata must name before/after scans, set same_environment_verified true and identify the real verification source. These fields are evidence supplied by a dataset curator; the converter cannot establish correspondence from unrelated scene IDs. An unchanged single scan is never advertised as a changed pair.

ScanNet `.sens` v4 extraction supports PNG/JPEG RGB and zlib-compressed uint16 depth; unsupported compression fails. Native calibration, camera-to-world poses, RGB/depth timestamps and metric depth conversion are preserved. ScanNet extraction does not download the dataset. 3RScan adapters preserve RGB calibration and available poses from `sequence/frame-*.color.jpg`, `_info.txt` and `.pose.txt`. 3RScan PGM depth is converted to metric NPY only when native depth shift, calibration and both sensor extrinsics are supplied; otherwise unavailable calibration is explicitly recorded. Compatible physical change-annotation conversion remains unavailable and explicitly recorded. Existing extracted ScanNet RGB sequences lacking sensor metadata record unavailable timestamps rather than claiming originals; encoded 30 fps schedules are labeled accordingly. Splits are retained from the explicitly supplied dataset split; no automatic split inference is claimed.

Ground-truth annotations validate scene ID, change class and optional position vectors. Metric positions require scale_verified. UNKNOWN differs from observed physical changes. Dataset object IDs and labels are not automatically converted into physical change labels. No real dataset conversion has yet been validated.
