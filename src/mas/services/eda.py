from __future__ import annotations

import pandas as pd

from mas.domain.models import EDASummary


def summarize_column(df: pd.DataFrame, column: str) -> EDASummary:
    s = df[column]
    null_ratio = float(s.isna().mean()) if len(df) else 0.0
    dtype = str(s.dtype)
    notes_parts: list[str] = []
    if pd.api.types.is_numeric_dtype(s):
        if s.notna().any():
            notes_parts.append(f"mean={s.mean():.4g}")
            notes_parts.append(f"std={s.std():.4g}")
    else:
        vc = s.astype(str).value_counts(dropna=True).head(5)
        notes_parts.append("top_values=" + ";".join(f"{i}:{v}" for i, v in vc.items()))
    return EDASummary(
        column=column,
        dtype=dtype,
        non_null_count=int(s.notna().sum()),
        null_ratio=null_ratio,
        notes=", ".join(notes_parts),
    )


def eda_all_columns(df: pd.DataFrame) -> dict[str, EDASummary]:
    return {c: summarize_column(df, c) for c in df.columns}
