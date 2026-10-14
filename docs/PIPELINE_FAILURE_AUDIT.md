# Current phone-pair failure audit

This audit covers the pink-container failure path in the saved phone pair. It is
not a claim that every repository line or supported dataset has been validated.

| Stage | Measured problem | Action / status |
|---|---|---|
| Crop recognition | Grounding DINO labels the tight container crop as table; four rotations do not fix it | Rotation recovery remains opt-in. Need broader context and validated semantic alternatives; raw maximum score is not an identity. |
| Localization | High raw point confidence previously admitted points inconsistent with their generating pixel | Fixed: positive camera depth, finite projection, strict native-pixel error below configured limit. Applied to fresh and cached perception and visibility support. |
| Shared registration | Sparse validation passes, but target projection remains displaced toward an adjacent bottle | Unresolved object-level correspondence. Registration RMSE alone cannot certify every object projection. |
| Tracking | Target lacks independent multi-view support and cross-recording identity | Unresolved. Do not merge lookalikes or treat missing matches as removal. |
| Visibility | Zero supported free views; ray-box uncertainty bounds unavailable | Unresolved. Missing detections do not establish empty space. |
| Reporting | All 53 replay decisions remain UNKNOWN | Report this failure, not detection accuracy. |

## Corrected localization

The old source observation accepted 2,692 pixels. Of those, 383 had greater than
2-pixel residual even in the smaller reconstruction grid. When the actual
native-image limit is applied, 708 remain and 1,984 are rejected. Maximum accepted
native reprojection error is 1.99956 pixels. This is a consistency check against
predicted camera geometry, not a ground-truth depth measurement or calibration.

The full cached comparison completed in 21.16 seconds. It recomputed localization,
association, visibility and consensus; it reused neural geometry and perception
outputs after existing video/content/configuration checks. It confirmed no changes.
See `reports/localization_reprojection_replay.json`.

Private reproducibility artifacts in this workspace:

- `/workspace/localization-audit-20261014/config.json`
- `/workspace/localization-audit-20261014/run.log`
- `/workspace/localization-audit-20261014/tests.log`
- `/workspace/localization-audit-20261014/overlay.jpg`
- `/workspace/spatialmemory-localization-fix-20261014/visualization/index.html`

Replay command (repository installed in the existing virtual environment):

```bash
OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 /workspace/spatialmemory-venv/bin/spatialmemory \
  '/workspace/attachments/a5ac41bf-942c-4fce-bf9d-f376005009cc/before room.mp4' \
  '/workspace/attachments/17db9d20-db27-4ef4-b195-933f2a812511/after  room.mp4' \
  --config /workspace/localization-audit-20261014/config.json \
  --output /workspace/spatialmemory-localization-fix-20261014
```

Use a fresh output directory for another replay. Raw media, crops, masks, and
model weights are not committed. Tests: 327 passed, one dependency warning.
