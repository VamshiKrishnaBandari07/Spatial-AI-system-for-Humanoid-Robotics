"""Filesystem locations for model caches, pinned third-party sources and run output.

The defaults are the historical ``/workspace`` layout when that directory exists
and is writable, so existing installs keep working unchanged; otherwise they fall
back to ``~/.spatialmemory`` so an ordinary user can run the application without
creating ``/workspace``. Every location can also be overridden through an
environment variable (Docker, Colab and custom layouts). Resolution happens at call time, so setting a variable
before models are loaded takes effect.

Overrides change only *where* files are looked up. Pinned source commits,
checkpoint revisions and SHA-256 verification are unaffected.
"""

from __future__ import annotations

import os
from pathlib import Path

DEFAULT_WORKSPACE = "/workspace"


def _fallback_root() -> Path:
    """Per-user location used when the historical /workspace is unavailable."""
    return Path.home() / ".spatialmemory"


def workspace_dir() -> Path:
    """Root for everything: ``SPATIALMEMORY_WORKSPACE``, else /workspace if usable, else ~/.spatialmemory."""
    explicit = os.environ.get("SPATIALMEMORY_WORKSPACE")
    if explicit:
        return Path(explicit)
    legacy = Path(DEFAULT_WORKSPACE)
    if legacy.is_dir() and os.access(legacy, os.W_OK | os.X_OK):
        return legacy
    return _fallback_root()


def _resolve(variable: str, folder: str, writable: bool) -> Path:
    explicit = os.environ.get(variable)
    if explicit:
        return Path(explicit)
    if os.environ.get("SPATIALMEMORY_WORKSPACE"):
        return workspace_dir() / folder
    # An existing /workspace install keeps working even when mounted read-only
    # (models only need reading); output directories must be writable.
    legacy = Path(DEFAULT_WORKSPACE) / folder
    if legacy.is_dir() and (not writable or os.access(legacy, os.W_OK | os.X_OK)):
        return legacy
    return workspace_dir() / folder


def models_dir() -> Path:
    """Hugging Face / checkpoint cache (``SPATIALMEMORY_MODELS``)."""
    return _resolve("SPATIALMEMORY_MODELS", "spatialmemory-models", writable=False)


def thirdparty_dir() -> Path:
    """Pinned third-party source checkouts such as VGGT and SAM 2 (``SPATIALMEMORY_THIRDPARTY``)."""
    return _resolve("SPATIALMEMORY_THIRDPARTY", "spatialmemory-thirdparty", writable=False)


def runs_dir() -> Path:
    """Where the application stores completed runs (``SPATIALMEMORY_RUNS``)."""
    return _resolve("SPATIALMEMORY_RUNS", "spatialmemory-runs", writable=True)


def configure_hf_cache() -> None:
    """Point Hugging Face caches inside the model directory unless already set."""
    os.environ.setdefault("HF_HOME", str(models_dir() / "hf-home"))
    os.environ.setdefault("HF_XET_CACHE", str(models_dir() / "xet"))
