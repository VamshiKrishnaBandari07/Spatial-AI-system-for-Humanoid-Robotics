# Dataset workspace audit

Date: 2026-10-10. Recomputed from actual on-disk files. No source assets deleted or redownloaded.

| Dataset | Files | Storage MiB | Directory |
| --- | ---: | ---: | --- |
| arkitscenes | 17288 | 1101.9 | `data/datasets/arkitscenes` |
| seven_scenes | 3067 | 964.9 | `data/datasets/seven_scenes` |
| replica | 10 | 4276.8 | `data/datasets/replica` |
| 3rscan | 0 | 0.0 | `data/datasets/3rscan` |

CPU affinity: 5; exposed nvidia-smi: False; free disk: 15.65 GiB.

Replica storage includes an incomplete 4 GiB compressed prefix and one native atlas. It is not a complete scene. Duplicate prepared ARKit manifests describe different preparations of the same recording, not extra scenes.

All non-ancillary ZIP members were read to EOF with CRC verification. Windows thumbnail caches are explicitly excluded and left untouched. Local SHA-256 matches recorded acquisition journals where available; no upstream-published checksum claim is made. See `official_download_sources.json` for original URLs, retained archive paths, sizes and hashes.

Existing adapters: `src/spatialmemory/public_data.py`, `src/spatialmemory/datasets.py`. Annotation schemas: `src/spatialmemory/inputs.py`. Existing evaluator: `spatialmemory_x evaluate`; provenance-gated evaluator: `scripts/benchmark_scene.py`. Tests: `tests/`.

Executed command: `python scripts/audit_datasets.py --data-root data/datasets --output reports`. Native-frame checks and actual test outcomes are recorded in `reconstruction_input_validation.json` and `dataset_quality_report.md`. No current real before/after annotation benchmark exists in the audited assets.
