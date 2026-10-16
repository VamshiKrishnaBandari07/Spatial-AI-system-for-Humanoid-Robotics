from pathlib import Path

import pytest

from spatialmemory import paths

VARS = ["SPATIALMEMORY_WORKSPACE", "SPATIALMEMORY_MODELS", "SPATIALMEMORY_THIRDPARTY",
        "SPATIALMEMORY_RUNS", "HF_HOME", "HF_XET_CACHE"]


@pytest.fixture(autouse=True)
def clean_env(monkeypatch):
    for name in VARS:
        monkeypatch.delenv(name, raising=False)


@pytest.fixture
def fake_home(monkeypatch, tmp_path):
    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: home))
    return home


def test_usable_legacy_workspace_keeps_historical_layout(monkeypatch, tmp_path, fake_home):
    legacy = tmp_path / "workspace"
    legacy.mkdir()
    monkeypatch.setattr(paths, "DEFAULT_WORKSPACE", str(legacy))
    assert paths.models_dir() == legacy / "spatialmemory-models"
    assert paths.thirdparty_dir() == legacy / "spatialmemory-thirdparty"
    assert paths.runs_dir() == legacy / "spatialmemory-runs"


def test_missing_legacy_workspace_falls_back_to_user_directory(monkeypatch, tmp_path, fake_home):
    monkeypatch.setattr(paths, "DEFAULT_WORKSPACE", str(tmp_path / "does-not-exist"))
    assert paths.runs_dir() == fake_home / ".spatialmemory" / "spatialmemory-runs"
    assert paths.models_dir() == fake_home / ".spatialmemory" / "spatialmemory-models"


def test_unwritable_legacy_workspace_falls_back_for_runs_but_keeps_readonly_models(monkeypatch, tmp_path, fake_home):
    import os
    legacy = tmp_path / "workspace"
    (legacy / "spatialmemory-models").mkdir(parents=True)  # preinstalled, read-only mount
    legacy.chmod(0o555)
    (legacy / "spatialmemory-models").chmod(0o555)
    monkeypatch.setattr(paths, "DEFAULT_WORKSPACE", str(legacy))
    try:
        if os.access(legacy, os.W_OK):  # running as root: permissions cannot be enforced
            pytest.skip("cannot make a directory unwritable for this user")
        assert paths.models_dir() == legacy / "spatialmemory-models"
        assert paths.runs_dir() == fake_home / ".spatialmemory" / "spatialmemory-runs"
    finally:
        legacy.chmod(0o755)
        (legacy / "spatialmemory-models").chmod(0o755)


def test_app_import_works_for_user_without_workspace(monkeypatch, tmp_path, fake_home):
    """The documented launch command must not require creating /workspace."""
    import importlib
    monkeypatch.setattr(paths, "DEFAULT_WORKSPACE", str(tmp_path / "does-not-exist"))
    from spatialmemory import app

    reloaded = importlib.reload(app)
    try:
        assert reloaded.ROOT == (fake_home / ".spatialmemory" / "spatialmemory-runs").resolve()
        assert reloaded.ROOT.is_dir()
    finally:
        monkeypatch.undo()
        importlib.reload(app)


def test_workspace_override_moves_all_locations(monkeypatch, tmp_path):
    monkeypatch.setenv("SPATIALMEMORY_WORKSPACE", str(tmp_path))
    assert paths.models_dir() == tmp_path / "spatialmemory-models"
    assert paths.thirdparty_dir() == tmp_path / "spatialmemory-thirdparty"
    assert paths.runs_dir() == tmp_path / "spatialmemory-runs"


def test_specific_override_beats_workspace(monkeypatch, tmp_path):
    monkeypatch.setenv("SPATIALMEMORY_WORKSPACE", str(tmp_path / "ws"))
    monkeypatch.setenv("SPATIALMEMORY_MODELS", str(tmp_path / "models"))
    assert paths.models_dir() == tmp_path / "models"
    assert paths.thirdparty_dir() == tmp_path / "ws" / "spatialmemory-thirdparty"


def test_empty_override_is_ignored(monkeypatch, tmp_path, fake_home):
    monkeypatch.setattr(paths, "DEFAULT_WORKSPACE", str(tmp_path / "does-not-exist"))
    monkeypatch.setenv("SPATIALMEMORY_MODELS", "")
    assert paths.models_dir() == fake_home / ".spatialmemory" / "spatialmemory-models"


def test_hf_cache_follows_models_dir(monkeypatch, tmp_path):
    monkeypatch.setenv("SPATIALMEMORY_MODELS", str(tmp_path))
    paths.configure_hf_cache()
    import os
    assert os.environ["HF_HOME"] == str(tmp_path / "hf-home")
    assert os.environ["HF_XET_CACHE"] == str(tmp_path / "xet")
    monkeypatch.delenv("HF_HOME"); monkeypatch.delenv("HF_XET_CACHE")


def test_hf_cache_does_not_override_explicit_user_setting(monkeypatch, tmp_path):
    monkeypatch.setenv("HF_HOME", "/custom/hf")
    monkeypatch.setenv("SPATIALMEMORY_MODELS", str(tmp_path))
    paths.configure_hf_cache()
    import os
    assert os.environ["HF_HOME"] == "/custom/hf"
    monkeypatch.delenv("HF_XET_CACHE", raising=False)


def test_no_hardcoded_workspace_paths_in_package():
    """New code must use spatialmemory.paths instead of literal /workspace paths."""
    root = Path(paths.__file__).parent
    offenders = [p.name for p in root.glob("*.py")
                 if p.name != "paths.py" and "/workspace" in p.read_text()]
    assert offenders == []
