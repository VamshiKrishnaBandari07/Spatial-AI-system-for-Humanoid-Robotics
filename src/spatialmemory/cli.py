import argparse, json
from pathlib import Path
from .contracts import Config
from .pipeline import run


def main():
    parser = argparse.ArgumentParser(
        description="SpatialMemory-X evidence-first spatial analysis"
    )
    parser.add_argument("before", type=Path)
    parser.add_argument("after", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--config", type=Path)
    args = parser.parse_args()
    config = (
        Config.model_validate_json(args.config.read_text()) if args.config else Config()
    )
    report = run(
        args.before,
        args.after,
        args.output,
        config,
        progress=lambda s: print(s, flush=True),
    )
    print(
        json.dumps(
            {
                "output": str(args.output),
                "events": len(report["events"]),
                "registration": report["registration"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
