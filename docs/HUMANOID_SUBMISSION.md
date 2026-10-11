# Humanoid challenge: reproduction and evidence

Deadline: **16 October 2026, 23:59 BST** (22:59 UTC). This is a research prototype, **not a completed challenge submission**. Genuine user-provided phone-video execution and reliable semantic change detection remain unverified.

## What works and what does not

- Two RGB-video paths or a validated scene manifest enter the same pipeline; annotations and optional sensors are not required to run.
- Frame selection, actual VGGT reconstruction, segmentation/detection/appearance inference, registration gates, semantic evidence aggregation, JSON, PLY, supporting masks and a self-contained scene viewer are implemented.
- The approved real RGB-D pair completes source-assisted alignment. This does not demonstrate autonomous RGB-only shared registration.
- The semantic recovery repeat produces 28 UNKNOWN decisions and misses the supported native MOVED instance. No physical change was confirmed. The earlier twelve-view run produced 69 UNKNOWN decisions.
- 129 regressions pass, retaining all original 122. Synthetic association and displacement tests establish contract behavior, not real-world matching accuracy.
- Matching accuracy, full detection metrics and calibrated uncertainty are unavailable. The bounded movement mode keeps UNKNOWN when declared depth/alignment error budgets are missing. Scores are ranking affinities, not probabilities.

## Installation

From `/workspace/Spatial-AI-system-for-Humanoid-Robotics`:

```bash
bash scripts/install.sh
bash scripts/install_neural.sh
source /workspace/spatialmemory-venv/bin/activate
python -m pytest -q
```

FFmpeg and ffprobe must be installed. The scripts use existing pinned CPU dependencies, immutable model/source revisions and checksum checks; preserve their cache integrity. Models remain under `/workspace/spatialmemory-models`; private recordings and derived scenes stay outside Git. The current cloud instance has no GPU. These instructions reuse the verified installation workflow; fresh snapshot and Docker runtime restoration remain separate, unverified checks.

## Phone recording protocol

Record the same small indoor area twice with overlapping views of static walls, corners and furniture. Use a slow continuous walk, roughly 15–60 seconds per recording, steady lighting and visible texture. Avoid exclusively blank walls or close-up object shots. Cover each relevant object from multiple angles, including where a removed object previously stood and where an added object now stands. Keep motion reasonably stable; the videos need not share resolution, frame rate, duration or trajectory.

Capture an unchanged control pair and measured edits, including small and larger displacements, addition/removal and an occluded control. Have a reviewer record object identities and changes independently before seeing model predictions. Record actual distances only when measured; do not invent coordinates or label uncertain visibility as observed. Preserve unedited originals and timestamps. The original RGB-only challenge must use genuine phone recordings, not the 3RScan pair or synthetic perturbations.

## Exact RGB-only inference command

```bash
python -m spatialmemory_x run \
  --before /workspace/private-phone/before.mp4 \
  --after /workspace/private-phone/after.mp4 \
  --scene-id phone_scene_001 \
  --config configs/phone_challenge.json \
  --project-root /workspace/Spatial-AI-system-for-Humanoid-Robotics \
  --output /workspace/phone_scene_001_run
```

The two input paths are **placeholders for files the user must supply**, not claimed workspace assets. Choose a new output directory; preserve previous runs. A manifest is generated automatically as `scene_manifest.json`. `configs/phone_challenge.json` validates and selects VGGT plus the conservative multi-cue profile. It has no required depth, IMU, LiDAR or known poses. Its bounded movement decision intentionally abstains without quantified depth/alignment error budgets; declaring numbers without independent measurement does not validate them. The legacy movement rule remains configurable for comparison, not a calibrated replacement.

This command has not been executed on genuine user videos. The phone profile's four-frame neural selection may still fail on weak texture, repeated patterns, limited coverage or unsuitable camera paths. Registration failure stops geometric claims; no identity transform fallback makes a failed run successful. Monocular distances remain arbitrary reconstruction units unless scale is established.

## Outputs and viewer

Successful runs export `reconstruction/`, `changes/changes.json`, `reports/summary.json`, `diagnostics/`, supporting views/masks, and `visualization/index.html`. Open the self-contained viewer locally with the output directory intact. Rotate/zoom the shared clouds, toggle before/after layers, filter decisions and select an event to read its explanation and open supporting images. MOVED endpoints are connected; ADDED and REMOVED positions have separate colors. UNKNOWN remains visible. Changed-region PLY contains candidate locations, not independently verified changed surfaces.

The new generic viewer was generated and browser-tested on the actual approved-pair repeat at:

```text
/workspace/spatialmemory-semantic-recovery-final-20261010/visualization/index.html
```

This demonstrates export/visualization functionality, not a genuine smartphone demonstration. Restricted real scene visuals are local; they are not public GitHub artifacts. Existing permitted synthetic console screenshots in `docs/screenshots/` are explicitly synthetic evidence.

## Reproduce recovery diagnosis and real-pair evaluation

```bash
python scripts/audit_unknowns.py \
  --run /workspace/spatialmemory-3rscan-coverage-20261010 --output reports
python scripts/audit_appearance.py \
  --run /workspace/spatialmemory-3rscan-coverage-20261010 \
  --output reports/dinov2_mask_audit.json
python scripts/compare_association.py \
  --run /workspace/spatialmemory-3rscan-coverage-20261010 \
  --output reports/association_baselines.json
python scripts/analyze_movement.py --plan reports/3rscan_candidate_download_plan.json \
  --run /workspace/spatialmemory-3rscan-coverage-20261010 \
  --output reports/movement_uncertainty_analysis.json
PAIR_MANIFEST=data/datasets/3rscan/processed/02b33dfb-be2b-2d54-92d2-cd012b2b3c40__fcf66d9e-622d-291c-84c2-bb23dfe31327/manifest.json
python scripts/profile_inference.py --manifest "$PAIR_MANIFEST" \
  --config configs/3rscan_semantic_recovery.json --project-root . \
  --output /workspace/semantic_recovery_reproduction
MPLCONFIGDIR=/tmp/spatialmemory-matplotlib XDG_CACHE_HOME=/tmp/spatialmemory-font-cache \
python scripts/review_3rscan.py --plan reports/3rscan_candidate_download_plan.json \
  --run /workspace/spatialmemory-semantic-recovery-final-20261010 \
  --report-output reports/semantic_recovery_evaluation.json
node scripts/check_scene_viewer.cjs \
  /workspace/spatialmemory-semantic-recovery-final-20261010 \
  reports/scene_viewer_browser_check.json
```

The source dataset must already be obtained under official approval. The recovery configuration points to the preserved fresh VGGT run through the existing integrity-checked reuse mechanism; semantic models execute again. A new output is required for reproduction. Read `docs/3RSCAN_VALIDATION.md` for preparation, native transform conventions, train-split restrictions and annotation scope.

For supervised phone evaluation, independently review annotations, pair provenance and predicted/native identity mapping using `docs/annotation_protocol.md` and `scripts/benchmark_scene.py`. Its provenance gates must pass before publishing supervised metrics. Basic CLI classification arithmetic alone does not validate identities, full inventory coverage or annotation independence.

## Models, licensing and submission boundaries

Grounding DINO, DINOv2, SAM 2 and VGGT are pinned in existing installation/model code. `reports/dinov2_mask_audit.json` records the actual DINOv2 checkpoint hash and processor; the previous real-run reports retain model provenance. SAM 2 uses its image predictor with optical-flow consistency, not a demonstrated video-memory tracking system. MASt3R/COLMAP/SLAM/complete TSDF and independently calibrated probabilistic fusion remain selected candidates or incomplete components.

VGGT weights use CC-BY-NC-4.0; 3RScan requires official approved non-commercial access. Model code licensing does not grant dataset or commercial deployment rights. Do not commit private phone originals, restricted RGB/depth/PLY data, checkpoints or credentials. Check eligibility before sharing a company demonstration. Native ground-truth overlays must never be presented as model-detected changes.

## Completion gates

The regression suite, preserved metric alignment, diagnostic exports, conservative association implementation, real-pair repeat and 3D evidence viewer pass their scoped checks. **The challenge is incomplete:** the reviewed movement is missed, independent identity labels and error budgets are absent, genuine phone execution has not occurred, and no independent-scene accuracy or production-readiness claim is supported. See `reports/semantic_recovery_delivery.json` and `reports/semantic_recovery_files.json`.
