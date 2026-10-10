import argparse, json
from pathlib import Path
from .contracts import Config
from .inputs import load_manifest, Annotations
from .pipeline import run
from .evaluation import classification_metrics


def main():
    p = argparse.ArgumentParser(prog="spatialmemory_x")
    sub = p.add_subparsers(dest="command", required=True)
    r = sub.add_parser("run")
    r.add_argument("--before", type=Path)
    r.add_argument("--after", type=Path)
    r.add_argument("--manifest", type=Path)
    r.add_argument("--scene-id", default="direct_input")
    r.add_argument("--output", type=Path)
    r.add_argument("--config", type=Path)
    r.add_argument("--project-root", type=Path, default=Path.cwd())
    e = sub.add_parser("evaluate")
    e.add_argument("--manifest", type=Path, required=True)
    e.add_argument("--annotations", type=Path)
    e.add_argument("--report", type=Path)
    e.add_argument("--project-root", type=Path, default=Path.cwd())
    args = p.parse_args()
    try:
        if args.command == "run":
            config = (
                Config.model_validate_json(args.config.read_text())
                if args.config
                else Config()
            )
            metadata = None
            if args.manifest:
                if args.before or args.after:
                    raise ValueError("Choose manifest or direct video paths")
                manifest, metadata, warnings = load_manifest(
                    args.manifest, args.project_root
                )
                before = metadata["before"]["video"]
                after = metadata["after"]["video"]
                scene = manifest.scene_id
                for w in warnings:
                    print("WARNING:", w)
            else:
                if not args.before or not args.after:
                    raise ValueError("Both --before and --after are required")
                from .inputs import Manifest, Recording

                scene = Manifest(
                    scene_id=args.scene_id,
                    before=Recording(video=str(args.before)),
                    after=Recording(video=str(args.after)),
                ).scene_id
                before = args.before
                after = args.after
                manifest = Manifest(
                    scene_id=scene,
                    before=Recording(video=str(before.resolve())),
                    after=Recording(video=str(after.resolve())),
                )
            output = args.output or args.project_root / "results" / scene
            report = run(
                before,
                after,
                output,
                config,
                progress=lambda s: print(s, flush=True),
                scene_id=scene,
                input_metadata=metadata,
            )
            (output / "scene_manifest.json").write_text(
                manifest.model_dump_json(indent=2)
            )
            print(
                json.dumps(
                    {
                        "scene_id": scene,
                        "output": str(output),
                        "events": len(report["events"]),
                    }
                )
            )
        else:
            from .inputs import Manifest

            manifest = Manifest.model_validate_json(args.manifest.read_text())
            from .inputs import resolve

            annotation_path = args.annotations or (
                resolve(manifest.annotations, args.project_root)
                if manifest.annotations
                else None
            )
            if annotation_path is None or not annotation_path.exists():
                raise FileNotFoundError(
                    "Supervised evaluation requires real ground-truth annotations"
                )
            annotation = Annotations.model_validate_json(annotation_path.read_text())
            report_path = (
                args.report
                or args.project_root / "results" / manifest.scene_id / "report.json"
            )
            if not report_path.exists():
                raise FileNotFoundError("Run inference first or provide --report")
            report = json.loads(report_path.read_text())
            if (
                annotation.scene_id != manifest.scene_id
                or report["scene_id"] != manifest.scene_id
            ):
                raise ValueError("Scene mismatch")
            predicted = {
                e["object_id"]: e["change"]
                for e in report["events"]
                if e.get("object_id")
            }
            truth = [a.change_type.value for a in annotation.changes]
            pred = [predicted.get(a.object_id, "UNKNOWN") for a in annotation.changes]
            result = classification_metrics(truth, pred)
            result["missing_object_predictions"] = sum(
                a.object_id not in predicted for a in annotation.changes
            )
            result["warning"] = (
                "Evaluation matches exact object IDs. Associate local memory IDs with ground-truth IDs before evaluation; missing identities count as UNKNOWN. Unannotated predictions are reported separately and are not included in classification accuracy."
            )
            truth_ids = {a.object_id for a in annotation.changes}
            result["unannotated_object_predictions"] = sorted(
                set(predicted) - truth_ids
            )
            print(json.dumps(result, indent=2))
    except (ValueError, FileNotFoundError, FileExistsError) as exc:
        p.exit(2, f"Error: {exc}\n")
