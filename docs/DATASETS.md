# Dataset acquisition and integration — 2026-10-09

Dataset-only work. The existing model architecture, checkpoints, training strategy and running application were preserved. Raw and prepared files live under ignored `data/datasets/`; none are staged or published. Approximately 19.7 GiB disk remains free. [Machine-readable report](datasets_report.json) contains counts, sizes, modalities, provenance, checks and restrictions.

| Dataset | Verified local subset | Size on disk, including archives/preparation | Integration / restriction |
|---|---|---:|---|
| ARKitScenes | Two separate native visits: Training recording `47333462`, Validation recording `41069050`; 48 synchronized frames each | 1,102 MiB | Native RGB, millimetre depth converted to metres, per-frame intrinsics, original timestamps and camera-to-world poses validated. Variable intrinsics retained per frame. |
| 3RScan | No scan recordings downloaded | 0 | Official Terms of Use access approval required. Existing native reference/rescan adapter preserved. |
| Microsoft 7-Scenes | Heads `seq-02`, 1,000 native RGB-D/pose frames; 48 prepared | 965 MiB | One nested sequence fetched with HTTP byte ranges rather than the whole scene archive. RGB/depth are explicitly uncalibrated and unregistered; nominal KinectFusion depth intrinsics are preserved without inventing RGB calibration. |
| Replica | One native `apartment_0` HDR Ptex atlas, 5616×5616×3 float16 | 181 MiB | Incomplete scene asset. Tensor format and existing DINOv2 appearance input verified. Mesh, poses, renderings and a complete geometry scene are unavailable. |

ARKit native RGB/depth counts are 3,611/3,610 for Training and 1,900/1,899 for Validation. Each has one RGB frame without matching depth; it is excluded from synchronized preparation and listed in the JSON report. Depth confidence, high-resolution depth, IMU and real change annotations were not acquired. Apple source LICENSE, Microsoft MSR-LA RTF and Replica Research Terms are retained locally. Microsoft and Replica restrict use to noncommercial research/education; retain their citations and notices. 3RScan's repository code MIT license does not grant dataset access.

## Native sequence contract

`spatialmemory.public_data` supplies dataset-only adapters. A `sequence.json` represents one recording and has no invented before/after pair or change annotations. Each frame retains RGB/depth paths, native timestamps, per-frame camera parameters, units, alignment status and `T_world_camera`. RGB tensors are float32 CHW in [0,1]; depth tensors use metres with invalid pixels NaN. Camera convention is X right, Y down, Z forward. Native ARKit world-to-camera axis-angle poses are inverted exactly as in Apple's official loader.

ARKit matching uses native filenames and trajectory times, with pose error at most 5 ms and intrinsics error at most 1.1 ms. Timestamp matching happens before selecting 48 frames across the recording. Original files are immutable. Prepared MP4s use a documented carrier clock; source timestamps are retained in `timestamps.json` and passed to the existing ingestion layer. Measured source-frame sharpness is retained in the loading check.

The current core accepts one fixed camera. When native ARKit intrinsics vary, all calibration is preserved in `per_frame_intrinsics.json` and the native sequence loader, while the canonical recording does not advertise falsely fixed RGB calibration or aligned depth fusion. Poses and metric depth remain available to dataset consumers; this task does not change the existing model to fuse them.

Microsoft documents raw uncalibrated RGB and depth, and nominal depth intrinsics `(fx,fy,cx,cy)=(585,585,320,240)`. These are **not real RGB calibration**. Depth and RGB share native frame IDs but have no verified geometric alignment. Native pose rotations have small drift; the loader permits only a bounded residual, projects rotations to the nearest SO(3), and preserves original matrices and correction sizes. Maximum correction in the prepared subset was 0.000396 in Frobenius norm. Unsupported drift fails clearly. Hardware timestamps are unavailable; frame-index/30 Hz is explicitly derived.

The native 7-Scenes `Thumbs.db` has a bad CRC. This Windows thumbnail cache is excluded. All 3,000 required RGB/depth/pose members passed CRC validation, and repeat acquisition revalidated extracted data without overwriting or redownloading it.

## Splits and synthetic examples

`data/datasets/manifests/train.json`, `validation.json` and `test.json` each contain one prepared recording from a distinct source scene. ARKit Training visit 467138 is train; ARKit Validation visit 381644 is validation. The entire Heads scene is held out as a project test scene. Its original native train designation and `TrainSplit.txt`/`TestSplit.txt` remain intact: this project holdout is **not an official 7-Scenes test benchmark**. Synthetic derivatives stay with their parent training scene; split checks reject leakage even when the dataset name has a `synthetic_` prefix.

Seven controlled examples derive from the ARKit **training** scene: 0.20 m movement, 0.02 m movement, added cloned region, removed region, unchanged, lighting ×0.8, and image-space occlusion. They use sampled source surfaces and exact virtual geometric-region transformations. They are **not verified semantic objects or observed real-world changes**. Their JSON annotations, source scene, offsets, region definition, camera transforms, visible-region pixel counts and rendering method are recorded. Nearest-point projection leaves holes as invalid depth, not evidence of free space. Source viewpoints vary across eight frames per recording. Fixed synthetic intrinsics are explicitly derived from the first native calibration.

Each generated pair has a canonical `manifest.json`, aligned synthetic RGB/depth, poses, timestamp sidecars, region-ID maps, `changes.json` and `transformation.json`. All seven pass the existing manifest, depth, pose and annotation readers. The training split references `arkitscenes/augmentations/47333462/examples.json`. These examples support controlled geometric and input-layer experiments; model-predicted object IDs still need explicit ground-truth association for object-level evaluation.

Replica additionally has three exact photometric transformations of one atlas crop. Atlas patches are texture data, not camera recordings. No Replica geometric change examples are claimed without the missing scene assets.

## Commands

Activate `/workspace/spatialmemory-venv/bin/activate` in the repository first.

```bash
python scripts/datasets.py download-arkit --video-id 47333462 --native-split Training --output data/datasets/arkitscenes/raw
python scripts/datasets.py download-arkit --video-id 41069050 --native-split Validation --output data/datasets/arkitscenes/raw
python scripts/datasets.py download-seven-scenes --scene heads --sequence seq-02 --output data/datasets/seven_scenes/raw
python scripts/datasets.py prepare-arkit --source data/datasets/arkitscenes/raw/Training/47333462 --scene-id arkit_visit_467138 --recording-id 47333462 --split train --output NEW_OUTPUT --max-frames 48
python scripts/datasets.py prepare-seven-scenes --source data/datasets/seven_scenes/raw/heads/seq-02/seq-02 --scene-id heads --recording-id seq-02 --split test --camera data/datasets/seven_scenes/camera_depth_nominal.json --output NEW_OUTPUT --max-frames 48
python scripts/datasets.py validate-replica-atlas data/datasets/replica/raw/apartment_0/textures/27-color-ptex.hdr
python scripts/augment_dataset_geometry.py --sequence data/datasets/arkitscenes/processed/47333462_synchronized/sequence.json --output NEW_AUGMENTATION_OUTPUT
```

Use new output directories for preparation; existing data is preserved. Downloads verify local integrity journals before reuse, reject oversize assets, retain interrupted partial files and enforce disk reserves. Archives reject path escapes, symlinks and excessive expanded sizes. HTTPS/TLS and CRC checks remain enabled. Local SHA-256 journals detect corruption but are not upstream-published checksums.

## Executed checks and remaining access steps

67 tests passed, including nine dataset tests. Real ARKit tensor/depth/K/pose loading passed; world-camera projection round-trip residual was approximately 1.2e-13 pixels. This measures internal numerical consistency, not sensor accuracy. Twelve real source frames passed the existing ingestion layer with original timestamps and pose synchronization. Real 7-Scenes tensor/depth/pose loading passed. The actual Replica atlas crop produced a finite normalized 384-dimensional descriptor through the existing DINOv2 expert. All seven synthetic pair manifests passed canonical readers. No training was started and no real change-detection accuracy was measured.

For 3RScan, complete the official [Terms of Use form](https://forms.gle/NvL5dvB4tSFrHfQH6), obtain the approved downloader/access, and select a documented reference/rescan pair with its native split and change metadata. No request was submitted on your behalf. The public example host also returned proxy 403; no access restriction was bypassed.

Replica's official release is a monolithic gzip stream split into seventeen 2 GB parts. A bounded official range stream extracted one complete native atlas member. Fetching the full collection exceeds the intended subset/storage budget. A complete authorized individual-scene source or sufficient space for the official archive is needed before mesh rendering and geometric experiments can be validated.

Dataset host additions were saved to the environment draft: `docs-assets.developer.apple.com`, `www.microsoft.com`, `download.microsoft.com`, `waldjohannau.github.io` and `campar.in.tum.de`. Apple/Microsoft/project-page reads now succeed in this instance. Review/save the draft and publish the environment to retain the configuration for future tasks; publication and fresh-task restoration are not claimed.

## Official Replica releases follow-up — 2026-10-10

The requested [official releases page](https://github.com/facebookresearch/Replica-Dataset/releases) was checked, and its v1.0 split archive used directly. A bounded 4 GiB compressed-prefix download did not reach the selected `apartment_0` geometry members. The prefix is retained under ignored `data/datasets/replica/cache/official_release_prefix.gz.partial` for reuse; it is **not a complete archive or verified geometry dataset**. `geometry_download.json` records the failure and missing members. The previously verified Ptex atlas and appearance integration remain available; zero complete scenes and zero camera recordings are claimed. Approximately 15.5 GiB remains free after caching. No full collection download or model changes were made.

`scripts/download_replica_subset.py` preserves existing assets, resumes through cached compressed bytes, enforces transfer/storage limits, and records incomplete downloads. Full gzip-footer integrity cannot be verified from a prefix; complete selected assets require independent native-format validation before integration.

The supplied [Glocker et al. (2013) bibliography](references/real-time-rgb-d-camera-relocalization.bib) is preserved as reference material for camera relocalization. It supplies a paper citation, not dataset files, calibration, annotations or a request to change the existing model.
