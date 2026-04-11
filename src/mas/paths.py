from __future__ import annotations

from pathlib import Path


def repo_root() -> Path:
    """Корень репозитория: .../src/mas/paths.py -> parents[2]."""
    return Path(__file__).resolve().parents[2]


def data_dir() -> Path:
    return repo_root() / "data"


def output_dir() -> Path:
    return repo_root() / "output"
