"""Recompute on-disk inventory and archive integrity without downloads or deletions."""

import argparse
from collections import Counter
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil
import subprocess
import zipfile
from spatialmemory.benchmarking import sha256


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=Path("data/datasets"))
    parser.add_argument("--output", type=Path, default=Path("reports"))
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    datasets, archives, sources = [], [], []
    for name in ["arkitscenes", "seven_scenes", "replica", "3rscan"]:
        folder = args.data_root / name
        files = sorted(p for p in folder.rglob("*") if p.is_file())
        datasets.append(
            {
                "dataset": name,
                "path": str(folder),
                "file_count": len(files),
                "size_bytes": sum(p.stat().st_size for p in files),
                "extensions": dict(Counter(p.suffix for p in files)),
                "sequence_manifests": [
                    str(p)
                    for p in files
                    if p.name == "sequence.json" and "augmentations" not in p.parts
                ],
            }
        )
        for archive in [p for p in files if p.suffix == ".zip"]:
            digest = sha256(archive)
            journal_path = archive.with_name(archive.name + ".download.json")
            journal = (
                json.loads(journal_path.read_text()) if journal_path.exists() else {}
            )
            verified = 0
            excluded = []
            with zipfile.ZipFile(archive) as zipped:
                for member in zipped.infolist():
                    if member.is_dir():
                        continue
                    if Path(member.filename).name in ["Thumbs.db", ".DS_Store"]:
                        excluded.append(member.filename)
                        continue
                    with zipped.open(member) as stream:
                        while stream.read(1024 * 1024):
                            pass
                    verified += 1
            if journal.get("sha256") and journal["sha256"] != digest:
                raise ValueError(f"Archive changed since acquisition: {archive}")
            entry = {
                "path": str(archive),
                "bytes": archive.stat().st_size,
                "sha256": digest,
                "members_crc_verified": verified,
                "excluded_ancillary_members": excluded,
                "acquisition_journal_hash_matches": journal.get("sha256") == digest,
                "upstream_checksum_verified": False,
            }
            archives.append(entry)
            sources.append(
                {
                    **entry,
                    "source_url": journal.get("source"),
                    "native_archive_member": journal.get("member"),
                    "journal": str(journal_path) if journal_path.exists() else None,
                }
            )
    usage = shutil.disk_usage(args.data_root)
    environment = {
        "cpu_affinity_count": len(os.sched_getaffinity(0)),
        "nvidia_smi_available": bool(shutil.which("nvidia-smi")),
        "free_disk_bytes": usage.free,
        "memory": subprocess.check_output(["free", "-h"], text=True),
    }
    date = datetime.now(timezone.utc).date().isoformat()
    inventory = {
        "date": date,
        "environment": environment,
        "datasets": datasets,
        "archives": archives,
        "3RScan": {
            "files": datasets[-1]["file_count"],
            "access": "official approval must be independently verified",
        },
    }
    (args.output / "dataset_inventory.json").write_text(
        json.dumps(inventory, indent=2) + "\n"
    )
    (args.output / "official_download_sources.json").write_text(
        json.dumps(sources, indent=2) + "\n"
    )
    lines = [
        "# Dataset workspace audit",
        "",
        f"Date: {date}. Recomputed from actual on-disk files. No source assets deleted or redownloaded.",
        "",
        "| Dataset | Files | Storage MiB | Directory |",
        "| --- | ---: | ---: | --- |",
    ]
    lines += [
        f"| {d['dataset']} | {d['file_count']} | {d['size_bytes'] / 1024**2:.1f} | `{d['path']}` |"
        for d in datasets
    ]
    lines += [
        "",
        f"CPU affinity: {environment['cpu_affinity_count']}; exposed nvidia-smi: {environment['nvidia_smi_available']}; free disk: {usage.free / 1024**3:.2f} GiB.",
        "",
        "Replica storage includes an incomplete 4 GiB compressed prefix and one native atlas. It is not a complete scene. Duplicate prepared ARKit manifests describe different preparations of the same recording, not extra scenes.",
        "",
        "All non-ancillary ZIP members were read to EOF with CRC verification. Windows thumbnail caches are explicitly excluded and left untouched. Local SHA-256 matches recorded acquisition journals where available; no upstream-published checksum claim is made. See `official_download_sources.json` for original URLs, retained archive paths, sizes and hashes.",
        "",
        "Existing adapters: `src/spatialmemory/public_data.py`, `src/spatialmemory/datasets.py`. Annotation schemas: `src/spatialmemory/inputs.py`. Existing evaluator: `spatialmemory_x evaluate`; provenance-gated evaluator: `scripts/benchmark_scene.py`. Tests: `tests/`.",
        "",
        "Executed command: `python scripts/audit_datasets.py --data-root data/datasets --output reports`. Native-frame checks and actual test outcomes are recorded in `reconstruction_input_validation.json` and `dataset_quality_report.md`. No current real before/after annotation benchmark exists in the audited assets.",
    ]
    (args.output / "dataset_audit.md").write_text("\n".join(lines) + "\n")
    print(
        json.dumps(
            {
                "status": "passed",
                "archives_checked": len(archives),
                "members_crc_verified": sum(
                    a["members_crc_verified"] for a in archives
                ),
                "datasets": [
                    {"name": d["dataset"], "bytes": d["size_bytes"]} for d in datasets
                ],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
