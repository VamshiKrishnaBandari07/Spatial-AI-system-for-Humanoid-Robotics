import importlib.util
import json
from pathlib import Path
import zipfile
import ast
import subprocess

import pytest
from spatialmemory import colab_ui
from spatialmemory.contracts import Config

ROOT = Path(__file__).parents[1]
spec = importlib.util.spec_from_file_location(
    "fixture", ROOT / "scripts/synthetic_fixture.py"
)
fixture = importlib.util.module_from_spec(spec)
spec.loader.exec_module(fixture)


def test_notebook_refresh_fast_forwards_without_overwriting_local_files(tmp_path):
    def git(*args):
        return subprocess.run(
            ["git", *map(str, args)], check=True, capture_output=True, text=True
        )

    origin, seed, checkout = [
        tmp_path / name for name in ("origin", "seed", "checkout")
    ]
    git("init", "--bare", "--initial-branch=main", origin)
    git("init", "--initial-branch=main", seed)
    git("-C", seed, "config", "user.name", "Test")
    git("-C", seed, "config", "user.email", "test@example.invalid")
    (seed / "README.md").write_text("first")
    git("-C", seed, "add", ".")
    git("-C", seed, "commit", "-m", "first")
    git("-C", seed, "remote", "add", "origin", origin)
    git("-C", seed, "push", "origin", "main")
    git("clone", origin, checkout)
    (checkout / "local-recording.txt").write_text("preserve")
    (seed / "README.md").write_text("second")
    git("-C", seed, "commit", "-am", "second")
    git("-C", seed, "push", "origin", "main")
    notebook = json.loads((ROOT / "notebooks/SpatialMemory_Colab.ipynb").read_text())
    tree = ast.parse("".join(notebook["cells"][1]["source"]))
    node = next(
        n
        for n in tree.body
        if isinstance(n, ast.If) and ast.unparse(n.test) == "not REPO.exists()"
    )
    code = compile(ast.Module(body=[node], type_ignores=[]), "setup-refresh", "exec")
    namespace = {
        "REPO": checkout,
        "subprocess": subprocess,
        "checked": lambda *args: subprocess.run(args, check=True, capture_output=True),
    }
    exec(code, namespace)
    assert (checkout / "README.md").read_text() == "second"
    assert (checkout / "local-recording.txt").read_text() == "preserve"
    (checkout / "README.md").write_text("local change")
    with pytest.raises(RuntimeError, match="preserved"):
        exec(code, namespace)
    assert (checkout / "README.md").read_text() == "local change"


def test_colab_real_geometry_workflow(tmp_path):
    before = fixture.make_video(tmp_path / "before.mp4")
    after = fixture.make_video(tmp_path / "after.mp4")
    config = tmp_path / "config.json"
    config.write_text(Config(focal_px=600).model_dump_json())
    results = list(
        colab_ui.analyze(
            str(before), str(after), "Geometry diagnostic", tmp_path / "runs", config
        )
    )
    status, summary, files = results[-1]
    assert "completed" in status
    assert summary["registration"]["valid"]
    assert any("reconstruction" in row[0] for row in results)
    report = json.loads(Path(files[0]).read_text())
    assert report["config"]["geometry_backend"] == "sparse"
    assert not report["config"]["semantics_enabled"]
    assert Path(files[2]).is_file()
    with zipfile.ZipFile(files[1]) as archive:
        assert archive.testzip() is None
        assert "reconstruction/aligned_scene.ply" in archive.namelist()
        assert not any(name.endswith(".mp4") for name in archive.namelist())


def test_colab_archive_scope(tmp_path):
    output = tmp_path / "output"
    output.mkdir()
    (output / "report.json").write_text("{}")
    (output / "before").mkdir()
    (output / "before/private.mp4").write_bytes(b"private")
    secret = tmp_path / "secret.json"
    secret.write_text("private")
    (output / "before/external.json").symlink_to(secret)
    with zipfile.ZipFile(colab_ui.results_zip(output)) as archive:
        assert archive.namelist() == ["report.json"]


def test_temporal_diagnostics_are_exported(tmp_path):
    output = tmp_path / "output"
    (output / "diagnostics").mkdir(parents=True)
    (output / "diagnostics/before_temporal.jsonl").write_text('{"frame_index":0}\n')
    with zipfile.ZipFile(colab_ui.results_zip(output)) as archive:
        assert "diagnostics/before_temporal.jsonl" in archive.namelist()


def test_distinct_cli_runs_do_not_overwrite_another_archive(tmp_path):
    sentinel = tmp_path / "results.zip"
    sentinel.write_bytes(b"existing")
    archives = []
    for name in ["scene_a", "scene_b"]:
        output = tmp_path / name
        output.mkdir()
        (output / "report.json").write_text(json.dumps({"scene_id": name}))
        archives.append(colab_ui.results_zip(output))
    assert archives[0] != archives[1]
    assert sentinel.read_bytes() == b"existing"
    with zipfile.ZipFile(archives[0]) as archive:
        assert json.loads(archive.read("report.json"))["scene_id"] == "scene_a"


def test_colab_missing_input():
    with pytest.raises(ValueError, match="Before video"):
        next(colab_ui.analyze(None, None, "Neural + objects", "unused", "unused"))


def test_colab_failure_is_not_success(tmp_path, monkeypatch):
    video = tmp_path / "video.mp4"
    video.write_bytes(b"corrupt")

    def fail(before, after, output, *args, **kwargs):
        output.mkdir()
        (output / "failure.json").write_text('{"status":"failed"}')
        raise ValueError("Registration gate failed")

    monkeypatch.setattr(colab_ui, "run", fail)
    results = list(
        colab_ui.analyze(
            str(video),
            str(video),
            "Neural + objects",
            tmp_path / "runs",
            ROOT / "configs/phone_challenge.json",
        )
    )
    assert "failed" in results[-1][0]
    assert results[-1][1] is None
    assert len(results[-1][2]) == 1
    assert not colab_ui.ACTIVE.locked()


def test_notebook_cells_compile_without_saved_secrets():
    notebook = json.loads((ROOT / "notebooks/SpatialMemory_Colab.ipynb").read_text())
    for index, cell in enumerate(notebook["cells"]):
        if cell["cell_type"] == "code":
            compile("".join(cell["source"]), f"cell{index}", "exec")
            assert cell["outputs"] == []
            assert cell["execution_count"] is None


def test_launch_checks_setup_before_requesting_password(monkeypatch):
    import getpass

    def forbidden_prompt(*args, **kwargs):
        pytest.fail("Incomplete setup must not request a password")

    monkeypatch.setattr(getpass, "getpass", forbidden_prompt)
    monkeypatch.setattr(Path, "is_file", lambda self: False)
    notebook = json.loads((ROOT / "notebooks/SpatialMemory_Colab.ipynb").read_text())
    with pytest.raises(RuntimeError, match="Run Step 1"):
        exec("".join(notebook["cells"][2]["source"]), {})


def test_launch_restores_configuration_without_setup_globals(monkeypatch):
    import subprocess

    monkeypatch.setattr(Path, "is_file", lambda self: True)
    monkeypatch.setattr(
        subprocess, "run", lambda *a, **kw: subprocess.CompletedProcess(a, 0)
    )
    notebook = json.loads((ROOT / "notebooks/SpatialMemory_Colab.ipynb").read_text())
    source = "".join(notebook["cells"][2]["source"])
    namespace = {}
    exec(source.split("password = getpass.getpass")[0], namespace)
    assert namespace["PYTHON"] == "/content/spatialmemory-colab-venv/bin/python"
    assert str(namespace["REPO"]) == "/content/Spatial-AI-system-for-Humanoid-Robotics"
    assert "dict(ENV," not in source
