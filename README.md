# SpatialMemory-X Ultra

**Multi-View Consensus and Uncertainty-Aware 4D Semantic World Modeling for Indoor Scene Change Detection**

Compare two recordings of the same indoor space, inspect their aligned 3D reconstruction, and review object changes with supporting visual evidence. SpatialMemory-X Ultra provides an RGB-only baseline and an optional neural research profile, with a responsive desktop and mobile workspace.

![SpatialMemory-X Ultra comparison workspace](docs/screenshots/testing-console-desktop.png)

The current release has **92 passing Python tests** and two passing browser verification suites. The screenshots and completed neural smoke runs use synthetic inputs. Real-world detection accuracy and robot deployment readiness require separate evaluation.

## What runs now

- MP4/MOV/MKV ingestion through FFmpeg/OpenCV, exact presentation timestamps including VFR, orientation handling through the decoder, input hashes and per-frame quality metadata.
- Pydantic scene manifests, annotations and optional calibration, depth, pose and IMU readers. Missing optional files permit RGB inference; malformed supplied metadata fails clearly.
- SIFT correspondences, essential-matrix initialization, triangulation, PnP pose extension and robust point refinement with fixed cameras.
- Cross-recording Sim(3) RANSAC registration with one-to-one correspondences, detected-foreground exclusion and held-out correspondence checks. Registration failure blocks geometric claims.
- Sparse region candidates with explicit UNKNOWN when free space is unverified; JSON diagnostics and PLY exports.
- Local video upload app, actual job status, interactive 3D point/camera/object layers, change filters, supporting mask images, scene memory, registration diagnostics and mobile layout.
- Independently tested visibility, correlation-group likelihood fusion, identity ambiguity, immutable SQLite scene snapshots, camera-compensated optical flow and experimental grid A*.
- Native ScanNet v4 `.sens` extraction and 3RScan/ScanNet sequence adapters with explicit valid-pair metadata and retained splits. Adapter regression tests use synthetic native-format fixtures; accessible ARKitScenes and 7-Scenes data have separate verified ingestion checks.

The research profile connects object detection, box-prompted SAM 2 masks, masked DINOv2 appearance, registered geometry, distinct-view consensus, predicted-depth visibility and SQLite snapshots. Ambiguous identities and insufficient views remain UNKNOWN. Optical-flow mask propagation is a consistency diagnostic; SAM 2 video-memory tracking is not implemented. Neural free-space evidence produces probable decisions, not measured-depth verification. Full bundle adjustment, dense TSDF, calibrated uncertainty, floor-driven navigation and real-scene acceptance remain outstanding.

## Installation

Python 3.12 on Linux was validated. FFmpeg and ffprobe must be available.

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements/cpu.lock.txt
.venv/bin/python -m pip install --no-build-isolation --no-deps -e .
.venv/bin/python -m pytest -q
```

Cloud installation: `bash scripts/install.sh`. The prepared environment is `/workspace/spatialmemory-venv`; a separate clean installation is tested under `/workspace/spatialmemory-clean`. CPU lockfiles pin the validated dependency versions. Neural dependencies are optional; CPU torch installation uses its official verified index:

```bash
.venv/bin/python -m pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cpu
.venv/bin/python -m pip install -r requirements/neural.lock.txt
```

No checkpoint is bundled. Review [MODEL_REGISTRY.md](MODEL_REGISTRY.md) before obtaining weights. Neural experts require immutable model revisions and do not execute remote model Python code. VGGT uses reviewed official source at a pinned commit and a SHA-256 verified checkpoint. The original VGGT-1B checkpoint is CC-BY-NC-4.0: this profile is for noncommercial research. No checkpoint is redistributed.

## Run ordinary RGB videos

```bash
python -m spatialmemory_x run --before before.mp4 --after after.mov --scene-id room_01 --output results/room_01
python -m spatialmemory_x run --manifest examples/scene_manifest.json --project-root .
python -m spatialmemory_x evaluate --manifest examples/scene_manifest.json --annotations data/annotations/scene_001/changes.json
```

Activate the installed virtualenv before using these commands. `run` never needs annotations. `evaluate` needs actual ground truth and a completed report; missing object identities count as UNKNOWN. The manifest is an example and requires your own recordings. All manifest paths resolve against `--project-root` (default: current directory), including optional metadata paths. Resolved paths escaping that root are rejected. Direct CLI video paths are explicit and may reside elsewhere.

Outputs contain `report.json`, `world.npz`, `reconstruction/*.ply`, `changes/changes.json`, `changes/changed_regions.ply`, `reports/summary.json` and `diagnostics/*.json`. Selected overview frames and manifests reside under `before/` and `after/`; semantic runs additionally save actual masks and annotated object frames in `evidence/supporting_frames/`, plus `spatial_memory.json` and SQLite observations. Concurrent output writers are rejected and completed runs are not overwritten. Failures write `failure.json`. The current implementation records partial stages but does not resume them automatically.

## Application

```bash
python -m uvicorn spatialmemory.app:app --host 127.0.0.1 --port 8000
```

This is a local application. Cloud onboarding does not provide a public web preview. Uploaded videos are retained locally under `SPATIALMEMORY_RUNS` (default `/workspace/spatialmemory-runs`). No video is sent to model APIs. One active job and a 512 MiB limit per video bound concurrent processing. The app is not a multi-user authenticated deployment.

## Synthetic reproducibility check

```bash
python scripts/synthetic_fixture.py /tmp/spatialmemory-fixtures
python -m spatialmemory_x run --before /tmp/spatialmemory-fixtures/before.mp4 --after /tmp/spatialmemory-fixtures/after.mp4 --output /tmp/spatialmemory-demo
```

These generated videos depict textured panels at varying depths. They are **synthetic**, identical no-change controls and must not be presented as smartphone data or real-world accuracy. See [EVALUATION.md](EVALUATION.md) for measured test evidence and unrun acceptance tests.

## Scientific limitations

Monocular distances are arbitrary reconstruction units, not meters. Without calibration, focal length is a documented heuristic and its uncertainty is unquantified. Sparse reconstruction cannot prove a previously occupied region is free. An empty candidate list does not prove every object is unchanged. RANSAC assumes static background dominates matches; repeated textures or moved foreground can invalidate that assumption. Reported residuals measure correspondence fit, not calibrated registration covariance or independent scene accuracy. Relative neural depth cannot establish metric visibility without geometric calibration and validation.

See [ARCHITECTURE.md](ARCHITECTURE.md), [docs/INPUTS.md](docs/INPUTS.md), [PROGRESS.md](PROGRESS.md) and [FAILURE_CASES.md](FAILURE_CASES.md). Project-authored code uses MIT; third-party models and datasets retain their own terms. No weights, recordings or credentials are committed. See the [methodology review](docs/METHODOLOGY.md) for implementation status against the original specification.

## Verified neural research profile

`bash scripts/install_neural.sh` installs pinned CPU neural dependencies, verifies the existing official VGGT source commit and downloads pinned public checkpoints into `/workspace/spatialmemory-models`. The model cache needs writable HF_HOME/HF_XET_CACHE paths, set by the script. Large VGGT weights occupy about 5 GB. Two-frame VGGT loading/inference used about 9.7 GiB peak process RSS in the measured synthetic smoke; larger inputs need profiling.

```bash
python scripts/neural_smoke.py FRAME.jpg
python scripts/vggt_smoke.py FRAME_1.jpg FRAME_2.jpg
python -m spatialmemory_x run --before before.mp4 --after after.mp4 --scene-id room_01 --config configs/vggt_cpu.json --output results/room_01_vggt
```

The UI also offers a VGGT research geometry profile. Neural depth, camera estimates and raw confidence are model predictions; their accuracy/scale are unverified. The pipeline still requires cross-recording registration and keeps sparse surface differences UNKNOWN without positive free-space observations. Depth Anything remains an independent diagnostic expert; the semantic profile uses VGGT predicted surfaces, Grounding DINO, SAM 2 and DINOv2. Grounding DINO called part of the synthetic panel fixture a table at raw score 0.326, illustrating why scores are not calibrated evidence.

## Integrated object research profile

```bash
bash scripts/install_neural.sh
python -m spatialmemory_x run --before before.mp4 --after after.mp4 --scene-id room_01 --config configs/consensus_cpu.json --output results/room_01_consensus
```

The installer also verifies official SAM 2 source/checkpoint provenance. Source dependency checkouts and weights live outside the project tree. In the app choose VGGT and enable object masks. The bounded CPU profile uses two frames per recording; increase `neural_frames_per_video` only after checking available memory. Two frames may provide insufficient independent views for a decision.

A full browser upload with actual pretrained inference completed in 144.9 seconds on synthetic identical panels, returning two model-proposed UNCHANGED regions and zero physical changes. The proposed “electronic device” labels are not verified and do not describe known real objects. [Desktop screenshot](docs/screenshots/semantic-workspace.png) and [mobile screenshot](docs/screenshots/mobile-workspace.png) show this synthetic run. See [browser evidence](docs/browser_smoke.json).

Cloud browser QA requires Playwright and Chromium: `node scripts/check_browser.cjs` exercises uploads, inference, masks, controls, reload and mobile layout. Set `SPATIALMEMORY_EXISTING_JOB` to replay a completed run without recomputing geometry; this replay verifies UI behavior, not a new inference run. Geometry reuse through `geometry_reuse_from` checks source hashes, geometry configuration, pinned provenance and exact preprocessed frames before loading cached arrays.

Optional evidence likelihood fitting uses `scripts/fit_calibration.py` with supplied labeled records. The saved Beta-Bernoulli training audit is not deployment calibration: held-out real evidence is still required, and rule gates remain active.

## Downloaded research datasets

Dataset-only acquisition prepared two official ARKitScenes recordings and one 7-Scenes sequence, with scene-disjoint manifests and seven explicitly synthetic geometric-region pairs. Replica has a verified partial atlas asset; 3RScan requires official access approval. See [dataset status and commands](docs/DATASETS.md) and [verified inventory](docs/datasets_report.json). No training or architecture changes were made for dataset integration.

## Comparison workspace

The local UI on port 8000 now includes named comparisons, before/after previews, a guided comparison workflow, keyboard-accessible 3D controls, evidence review, recent comparisons and separate model-report/human-review exports. Reviews and comparison metadata stay in the current browser; they do not change inference results. See [testing workflow and verification](docs/TESTING_UI.md), [desktop preview](docs/screenshots/testing-console-desktop.png), and [mobile preview](docs/screenshots/testing-console-mobile.png).

## Real-data recovery and annotation

The current audit verifies seven retained archives (19,531 CRC-checked payload files), 144 prepared native RGB-D frames and one full neural inference control on real ARKit imagery. The control completed in 171.7 seconds with 9.7 GiB peak individual child RSS and an UNKNOWN object result. It is one recording supplied twice, not a two-visit change benchmark. Real-world accuracy and F1 remain unmeasured.

Open `/annotate` in the local app to review two recordings, associate objects, record uncertain labels, independently review and export schema-validated annotations. This page keeps videos in the browser. Ground-truth annotations are separate from comparison-dashboard assessments and model predictions. See the [annotation protocol](docs/annotation_protocol.md).

```bash
python scripts/audit_datasets.py --data-root data/datasets --output reports
python scripts/benchmark_scene.py \
  --manifest data/manifests/SCENE.json \
  --pair-evidence data/annotations/SCENE/pair_evidence.json \
  --annotations data/annotations/SCENE/changes.json \
  --report results/SCENE/report.json \
  --identity-mapping data/annotations/SCENE/identity_mapping.json \
  --project-root . --output reports/SCENE
```

These benchmark paths are placeholders for verified data, not included recordings or labels. The evaluator rejects unverified relationships, source/input hash mismatches, unreviewed or model-generated ground truth, and unreliable registration. Missing metrics remain unavailable; unannotated detections and UNKNOWN decisions stay visible. For actual memory measurements, use `scripts/profile_inference.py` and pass its matching `--resource-profile` to the benchmark.

[Dataset audit](reports/dataset_audit.md) · [capability matrix](reports/dataset_capability_matrix.csv) · [dataset selection](reports/dataset_selection.md) · [quality report](reports/dataset_quality_report.md) · [completion status](reports/completion_status.json). 3RScan/RIO10 require official approval. Other candidate hosts remain blocked by runtime policy; licences and small-pair availability must be verified before acquisition. No genuine annotated real change pair has been obtained, so the ten benchmark completion conditions remain unsatisfied.

## Dataset access preparation

See [official access steps and smartphone templates](docs/DATASET_ACCESS.md) for the remaining real-data requirements. Prepared empty directories are placeholders. TUM RGB-D acquisition is blocked by the current network policy; the required domain addition is saved for environment review and publication.

### Approved 3RScan subset

One official reference/rescan pair is now downloaded and integrated, following confirmed access approval. All 257 native RGB-D/pose frame sets and instance-labeled geometry passed validation. The canonical label draft remains unreviewed; supervised prediction accuracy is unavailable. See [acquisition](reports/3rscan_acquisition.json), [native validation](reports/3rscan_native_validation.json) and [exact paths and remaining review requirements](docs/DATASET_ACCESS.md). Raw files remain excluded from Git.
