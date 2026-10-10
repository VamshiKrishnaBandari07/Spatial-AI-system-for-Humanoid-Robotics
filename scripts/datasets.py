"""Dataset-only acquisition/preparation. Never starts training or changes model settings."""

import argparse
import json
from pathlib import Path

from spatialmemory.inputs import Camera
from spatialmemory.public_data import (
    arkit_sequence,
    seven_scenes_sequence,
    export_recording,
    download_asset,
    extract_zip,
    replica_atlas,
    write_splits,
    download_zip_member,
    verify_extracted_zip,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    a = sub.add_parser("download-arkit")
    a.add_argument("--video-id", required=True)
    a.add_argument("--native-split", choices=["Training", "Validation"], required=True)
    a.add_argument("--output", type=Path, required=True)
    a.add_argument("--max-asset-mib", type=int, default=512)
    seven = sub.add_parser("download-seven-scenes")
    seven.add_argument(
        "--scene",
        choices=["heads", "chess", "fire", "office", "pumpkin", "redkitchen", "stairs"],
        default="heads",
    )
    seven.add_argument("--sequence", default="seq-02")
    seven.add_argument("--output", type=Path, required=True)
    for name in ["prepare-arkit", "prepare-seven-scenes"]:
        p = sub.add_parser(name)
        p.add_argument("--source", type=Path, required=True)
        p.add_argument("--scene-id", required=True)
        p.add_argument("--recording-id", required=True)
        p.add_argument(
            "--split", choices=["train", "validation", "test"], required=True
        )
        p.add_argument("--output", type=Path, required=True)
        p.add_argument("--max-frames", type=int, default=48)
        if name == "prepare-seven-scenes":
            p.add_argument("--camera", type=Path, required=True)
    r = sub.add_parser("validate-replica-atlas")
    r.add_argument("path", type=Path)
    s = sub.add_parser("splits")
    s.add_argument("--sequence", type=Path, action="append", default=[])
    s.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        if args.command == "download-seven-scenes":
            import re

            if not re.fullmatch(r"seq-[0-9]{2}", args.sequence):
                raise ValueError("Invalid native sequence ID")
            base = "https://download.microsoft.com/download/2/8/5/28564B23-0828-408F-8631-23B1EFF1DAC8/"
            folder = args.output / args.scene
            for name in ["TrainSplit.txt", "TestSplit.txt", args.sequence + ".zip"]:
                path = download_zip_member(
                    base + args.scene + ".zip", args.scene + "/" + name, folder / name
                )
                if path.suffix == ".zip":
                    target = folder / args.sequence
                    if target.exists():
                        verify_extracted_zip(
                            path, target, exclude_names=("Thumbs.db", ".DS_Store")
                        )
                        continue
                    extract_zip(path, target, exclude_names=("Thumbs.db", ".DS_Store"))
            print(
                json.dumps(
                    {
                        "status": "downloaded; prepare with nominal depth calibration, no RGB alignment assumption",
                        "location": str(folder),
                    }
                )
            )
        elif args.command == "download-arkit":
            if not args.video_id.isdigit():
                raise ValueError("Native video ID must be numeric")
            if args.max_asset_mib < 1 or args.max_asset_mib > 1024:
                raise ValueError("Asset budget must be 1..1024 MiB")
            base = f"https://docs-assets.developer.apple.com/ml-research/datasets/arkitscenes/v1/raw/{args.native_split}/{args.video_id}"
            folder = args.output / args.native_split / args.video_id
            for asset in [
                "lowres_wide.zip",
                "lowres_depth.zip",
                "lowres_wide_intrinsics.zip",
                "lowres_wide.traj",
            ]:
                path = download_asset(
                    base + "/" + asset, folder / asset, args.max_asset_mib * 1024**2
                )
                if path.suffix == ".zip":
                    marker = folder / (asset + ".extracted.json")
                    if marker.exists():
                        continue
                    stage = folder / (asset + ".extraction")
                    extract_zip(path, stage)
                    # Native archives contain a named asset directory. Move only into a new path.
                    source = stage / asset.removesuffix(".zip")
                    if not source.is_dir():
                        source = stage
                    target = folder / asset.removesuffix(".zip")
                    if target.exists():
                        raise FileExistsError(target)
                    source.rename(target)
                    marker.write_text(
                        json.dumps({"archive": str(path), "directory": str(target)})
                    )
            print(
                json.dumps(
                    {
                        "status": "downloaded; preparation/validation still required",
                        "location": str(folder),
                    }
                )
            )
        elif args.command.startswith("prepare-"):
            kw = dict(
                root=args.source,
                scene_id=args.scene_id,
                recording_id=args.recording_id,
                split=args.split,
                max_frames=args.max_frames,
            )
            if args.command == "prepare-arkit":
                sequence = arkit_sequence(**kw)
            else:
                sequence = seven_scenes_sequence(
                    **kw, camera=Camera.model_validate_json(args.camera.read_text())
                )
            recording = export_recording(sequence, args.output)
            print(
                json.dumps(
                    {
                        "status": "validated and prepared",
                        "frames": len(sequence.frames),
                        "recording": recording.model_dump(),
                        "changes": None,
                    }
                )
            )
        elif args.command == "validate-replica-atlas":
            atlas = replica_atlas(args.path)
            print(
                json.dumps(
                    {
                        "shape": list(atlas.shape),
                        "dtype": str(atlas.dtype),
                        "finite": True,
                        "camera": None,
                        "depth": None,
                        "poses": None,
                        "recording": False,
                    }
                )
            )
        else:
            print(json.dumps(write_splits(args.sequence, args.output), indent=2))
    except (ValueError, FileExistsError, FileNotFoundError) as error:
        parser.exit(2, f"Error: {error}\n")


if __name__ == "__main__":
    main()
