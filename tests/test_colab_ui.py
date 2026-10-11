import importlib.util
import json
from pathlib import Path
import zipfile

import pytest
from spatialmemory import colab_ui
from spatialmemory.contracts import Config

ROOT = Path(__file__).parents[1]
spec = importlib.util.spec_from_file_location(
    "fixture", ROOT / "scripts/synthetic_fixture.py"
)
fixture = importlib.util.module_from_spec(spec)
spec.loader.exec_module(fixture)


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
