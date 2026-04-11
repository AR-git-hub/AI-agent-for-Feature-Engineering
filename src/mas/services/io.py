from __future__ import annotations

from pathlib import Path

import pandas as pd


def read_readme(data_dir: Path) -> tuple[Path | None, str]:
    p = data_dir / "readme.txt"
    if not p.is_file():
        return None, ""
    return p, p.read_text(encoding="utf-8")


def read_train_test(data_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    train = pd.read_csv(data_dir / "train.csv")
    test = pd.read_csv(data_dir / "test.csv")
    return train, test


def infer_id_and_target(train: pd.DataFrame, test: pd.DataFrame) -> tuple[str, str | None]:
    """Эвристика: target — колонка только в train; id — общая, предпочтительно 'id'."""
    train_cols = set(train.columns)
    test_cols = set(test.columns)
    only_train = train_cols - test_cols
    if len(only_train) == 1:
        target = next(iter(only_train))
    elif "target" in train_cols:
        target = "target"
    else:
        target = None

    common = [c for c in train.columns if c in test.columns]
    if "id" in common:
        id_col = "id"
    elif common:
        id_col = common[0]
    else:
        id_col = train.columns[0]
    return id_col, target
