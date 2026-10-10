from pathlib import Path
import numpy as np, json


def ply(path, points, colors=None):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    points = np.asarray(points).reshape(-1, 3)
    if colors is not None and not np.isfinite(np.asarray(colors)).all():
        raise ValueError("PLY colors must be finite")
    colors = (
        np.full((len(points), 3), 180, dtype=int)
        if colors is None
        else np.clip(np.asarray(colors) * 255, 0, 255).astype(int)
    )
    if colors.shape != points.shape:
        raise ValueError("PLY colors must match the number of 3D points")
    if not np.isfinite(points).all():
        raise ValueError("PLY points must be finite")
    with path.open("w") as f:
        f.write(
            f"ply\nformat ascii 1.0\nelement vertex {len(points)}\nproperty float x\nproperty float y\nproperty float z\nproperty uchar red\nproperty uchar green\nproperty uchar blue\nend_header\n"
        )
        for p, c in zip(points, colors):
            f.write(" ".join(map(str, [*p, *c])) + "\n")


def export_run(output, report, before, after, bcolors, acolors):
    output = Path(output)
    ply(output / "reconstruction/before.ply", before, bcolors)
    ply(output / "reconstruction/after.ply", after, acolors)
    ply(
        output / "reconstruction/aligned_scene.ply",
        np.concatenate([before, after]),
        np.concatenate([bcolors, acolors]),
    )
    for folder in [
        "changes",
        "reports",
        "diagnostics",
        "evidence/supporting_frames",
        "visualization",
    ]:
        (output / folder).mkdir(parents=True, exist_ok=True)
    (output / "changes/changes.json").write_text(
        json.dumps({"schema_version": "1.0", "events": report["events"]}, indent=2)
    )
    # Candidate centroids are not misrepresented as confirmed changed surfaces.
    positions = []
    for event in report["events"]:
        if event["change"] not in ["ADDED", "REMOVED", "MOVED"]:
            continue
        # Export candidate locations, including both endpoints of movement.
        for position in [event["before_position"], event["after_position"]]:
            if position is not None:
                positions.append(position)
    ply(output / "changes/changed_regions.ply", positions)
    (output / "reports/summary.json").write_text(json.dumps(report, indent=2))
    (output / "diagnostics/registration.json").write_text(
        json.dumps(report["registration"], indent=2)
    )
    (output / "diagnostics/processing.json").write_text(
        json.dumps(report["stages"], indent=2)
    )

    from .viewer import write_viewer

    write_viewer(output, report, before, after)
