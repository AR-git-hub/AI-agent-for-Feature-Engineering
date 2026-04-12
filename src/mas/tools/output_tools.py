"""Тулы агента вывода: сохранение train/test в output/."""
from __future__ import annotations

import re
import unicodedata
from pathlib import Path

import pandas as pd


def _to_ascii_col(name: str) -> str:
    """Приводит имя колонки к ASCII — защита от UnicodeDecodeError в CatBoost."""
    normalized = unicodedata.normalize("NFKD", str(name))
    ascii_only = normalized.encode("ascii", errors="replace").decode("ascii")
    cleaned = re.sub(r"[^A-Za-z0-9_]", "_", ascii_only)
    cleaned = re.sub(r"_+", "_", cleaned).strip("_")
    return cleaned or "feat"


def ensure_output_dir(output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)


def _encode_non_numeric(
    df: pd.DataFrame,
    reserved_cols: set[str],
) -> pd.DataFrame:
    """Кодирует нечисловые *признаки* в Categorical int-коды.

    reserved_cols — колонки которые не трогаем (id, target и все исходные).
    Гарантирует, что scoring.py получит только числовые фичи.
    """
    df = df.copy()
    for col in df.columns:
        if col in reserved_cols:
            continue
        if not pd.api.types.is_numeric_dtype(df[col]):
            df[col] = pd.Categorical(
                df[col].astype(str).fillna("__na__")
            ).codes.astype(float)
    return df


def save_submission(
    output_dir: Path,
    train: pd.DataFrame,
    test: pd.DataFrame,
    *,
    id_col: str | None = None,
    target_col: str | None = None,
    reserved_cols: set[str] | None = None,
) -> tuple[Path, Path]:
    ensure_output_dir(output_dir)
    # reserved = переданный набор ИЛИ минимальный из id+target
    _reserved = reserved_cols if reserved_cols is not None else {
        col for col in (id_col, target_col) if col
    }
    train = _encode_non_numeric(train, _reserved)
    test  = _encode_non_numeric(test,  _reserved)

    # Приводим имена колонок к ASCII — scoring.py читает CSV без явной кодировки,
    # и CatBoost C++ падает с UnicodeDecodeError на кириллических именах
    ascii_cols_train = {c: _to_ascii_col(c) for c in train.columns}
    ascii_cols_test  = {c: _to_ascii_col(c) for c in test.columns}
    train = train.rename(columns=ascii_cols_train)
    test  = test.rename(columns=ascii_cols_test)

    train_path = output_dir / "train.csv"
    test_path  = output_dir / "test.csv"

    # Явный open(str(...)) обходит баг pandas на Windows,
    # где Path-объект в to_csv вызывает OSError [Errno 22] Invalid argument.
    with open(str(train_path), "w", encoding="utf-8", newline="") as fh:
        train.to_csv(fh, index=False)
    with open(str(test_path), "w", encoding="utf-8", newline="") as fh:
        test.to_csv(fh, index=False)

    return train_path, test_path
