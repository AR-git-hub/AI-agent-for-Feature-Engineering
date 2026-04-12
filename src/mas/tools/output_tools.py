"""Тулы агента вывода: сохранение train/test в output/."""
from __future__ import annotations

from pathlib import Path

import pandas as pd


def ensure_output_dir(output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)


def save_submission(
    output_dir: Path,
    train: pd.DataFrame,
    test: pd.DataFrame,
) -> tuple[Path, Path]:
    ensure_output_dir(output_dir)
    train_path = output_dir / "train.csv"
    test_path = output_dir / "test.csv"
    train.to_csv(train_path, index=False)
    test.to_csv(test_path, index=False)
    return train_path, test_path
