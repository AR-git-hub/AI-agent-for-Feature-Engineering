from __future__ import annotations

import pandas as pd


def fill_missing_basic(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for col in out.columns:
        s = out[col]
        if pd.api.types.is_numeric_dtype(s):
            out[col] = s.fillna(s.median())
        else:
            mode = s.mode(dropna=True)
            fill = mode.iloc[0] if len(mode) else ""
            out[col] = s.fillna(fill)
    return out


def winsorize_percentiles(df: pd.DataFrame, lower: float = 0.01, upper: float = 0.99) -> pd.DataFrame:
    out = df.copy()
    num_cols = out.select_dtypes(include="number").columns
    for col in num_cols:
        lo = out[col].quantile(lower)
        hi = out[col].quantile(upper)
        out[col] = out[col].clip(lower=lo, upper=hi)
    return out


def basic_clean(df: pd.DataFrame) -> pd.DataFrame:
    return winsorize_percentiles(fill_missing_basic(df))
