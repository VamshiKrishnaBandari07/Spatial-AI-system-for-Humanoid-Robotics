"""Run existing inference in a fresh process and record wall time and child peak RSS."""

import argparse
import json
from pathlib import Path
import resource
import subprocess
import sys
import time
from spatialmemory.benchmarking import sha256


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.exit(
            2, "Choose a new output directory; existing results are preserved.\n"
        )
    command = [
        sys.executable,
        "-m",
        "spatialmemory_x",
        "run",
        "--manifest",
        str(args.manifest),
        "--config",
        str(args.config),
        "--project-root",
        str(args.project_root),
        "--output",
        str(args.output),
    ]
    start = time.perf_counter()
    result = subprocess.run(command)
    duration = time.perf_counter() - start
    rss = resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss
    mib = rss / (1024**2 if sys.platform == "darwin" else 1024)
    args.output.mkdir(parents=True, exist_ok=True)
    report = args.output / "report.json"
    profile = {
        "status": "passed" if result.returncode == 0 else "failed",
        "exit_code": result.returncode,
        "wall_seconds": duration,
        "peak_child_process_rss_mib": mib,
        "memory_method": "Fresh wrapper process resource.getrusage(RUSAGE_CHILDREN).ru_maxrss; maximum individual child RSS, not aggregate process-tree or GPU memory",
        "report_sha256": sha256(report) if report.exists() else None,
        "manifest_sha256": sha256(args.manifest),
        "config_sha256": sha256(args.config),
        "command": command,
    }
    (args.output / "resource_profile.json").write_text(
        json.dumps(profile, indent=2) + "\n"
    )
    print(json.dumps(profile, indent=2), flush=True)
    raise SystemExit(result.returncode)


if __name__ == "__main__":
    main()
