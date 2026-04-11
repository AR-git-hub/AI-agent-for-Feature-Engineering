from __future__ import annotations

import re

import numpy as np
import pandas as pd


def _safe_name(prefix: str, *parts: str) -> str:
    raw = "__".join([prefix, *parts])
    return re.sub(r"[^0-9a-zA-Z_]+", "_", raw)[:120]


def build_catalog_ids(train: pd.DataFrame, id_col: str, target_col: str | None) -> list[str]:
    """Детерминированный каталог безопасных признаков (id для LLM)."""
    skip = {id_col}
    if target_col:
        skip.add(target_col)
    ids: list[str] = []
    num_cols = [c for c in train.columns if c not in skip and pd.api.types.is_numeric_dtype(train[c])]
    for c in num_cols:
        ids.append(f"zscore:{c}")
    for i, a in enumerate(num_cols):
        for b in num_cols[i + 1 :]:
            ids.append(f"mul:{a}:{b}")
    if not ids:
        ids.append("synthetic:const")
    return ids


def apply_catalog_features(
    train: pd.DataFrame,
    test: pd.DataFrame,
    feature_ids: list[str],
    id_col: str,
    target_col: str | None,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, str]]:
    """
    Возвращает (train_aug, test_aug, id->output_column_name).
    Статистики только по train; к тесту применяются те же преобразования.
    """
    skip = {id_col}
    if target_col:
        skip.add(target_col)

    tr = train.copy()
    te = test.copy()
    name_by_id: dict[str, str] = {}

    for fid in feature_ids:
        parts = fid.split(":")
        if parts[0] == "synthetic" and len(parts) == 2 and parts[1] == "const":
            col = _safe_name("feat", "const")
            tr[col] = 1.0
            te[col] = 1.0
            name_by_id[fid] = col
        elif parts[0] == "zscore" and len(parts) == 2:
            c = parts[1]
            if c not in tr.columns or c in skip:
                continue
            if not pd.api.types.is_numeric_dtype(tr[c]):
                continue
            mu = float(tr[c].mean())
            sig = float(tr[c].std(ddof=0)) + 1e-9
            col = _safe_name("z", c)
            tr[col] = (tr[c].astype(float) - mu) / sig
            if c in te.columns:
                te[col] = (te[c].astype(float) - mu) / sig
            else:
                te[col] = np.nan
            name_by_id[fid] = col
        elif parts[0] == "mul" and len(parts) == 3:
            a, b = parts[1], parts[2]
            if a not in tr.columns or b not in tr.columns or a in skip or b in skip:
                continue
            if not (
                pd.api.types.is_numeric_dtype(tr[a]) and pd.api.types.is_numeric_dtype(tr[b])
            ):
                continue
            col = _safe_name("mul", a, b)
            tr[col] = tr[a].astype(float) * tr[b].astype(float)
            if a in te.columns and b in te.columns:
                te[col] = te[a].astype(float) * te[b].astype(float)
            else:
                te[col] = np.nan
            name_by_id[fid] = col

    return tr, te, name_by_id
