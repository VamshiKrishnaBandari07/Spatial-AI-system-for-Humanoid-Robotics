"""Prepare an explicit, integrity-bound metric seed from approved native metadata."""

import argparse
import json
from pathlib import Path
from spatialmemory.benchmarking import sha256
from spatialmemory.native_review import review_native
from spatialmemory.inputs import load_manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    args = parser.parse_args()
    plan = json.loads(args.plan.read_text())
    review = review_native(plan)
    manifest, files, _ = load_manifest(args.manifest, Path.cwd())
    if manifest.scene_id != review["scene_id"]:
        raise ValueError("Manifest/native relationship mismatch")
    seed = {
        "unit": "metres",
        "source_frame": "rescan_world",
        "destination_frame": "reference_world",
        "T_reference_rescan": review["T_reference_rescan"],
        "relationship_file": plan["relationship_metadata_path"],
        "relationship_sha256": plan["relationship_metadata_sha256"],
        "before_recording_id": plan["reference"],
        "after_recording_id": plan["rescan"],
        "before_video_sha256": sha256(files["before"]["video"]),
        "after_video_sha256": sha256(files["after"]["video"]),
        "source_code_commit": "12f040d3dc849e394ed366acd7b54e63acb49205",
        "source_convention_evidence": "Official C++ data.cc ReadMatrix (Eigen column-major), rio.cc TransformPly2Reference, sequence.cc GetPose (metre transform multiplied by 1000 only when mm requested)",
    }
    for name, payload in [
        ("verified_initial.json", seed),
        ("evaluation_review.json", review),
    ]:
        path = args.manifest.parent / name
        if path.exists():
            current = json.loads(path.read_text())
            keys = (
                list(payload)
                if name == "evaluation_review.json"
                else [
                    "unit",
                    "source_frame",
                    "destination_frame",
                    "T_reference_rescan",
                    "relationship_file",
                    "relationship_sha256",
                    "before_recording_id",
                    "after_recording_id",
                    "before_video_sha256",
                    "after_video_sha256",
                ]
            )
            if any(current.get(key) != payload[key] for key in keys):
                raise FileExistsError(
                    f"Preserve existing differing preparation: {path}"
                )
        if not path.exists():
            path.write_text(json.dumps(payload, indent=2) + "\n")
    print(args.manifest.parent / "verified_initial.json")


if __name__ == "__main__":
    main()
