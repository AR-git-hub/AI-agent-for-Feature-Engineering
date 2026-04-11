from __future__ import annotations

import pandas as pd

from mas.domain.models import FeatureConfig


def stub_feature_config(column: str, df: pd.DataFrame) -> FeatureConfig:
    series = df[column] if column in df.columns else None
    n = int(series.nunique(dropna=True)) if series is not None else 0
    return FeatureConfig(column=column, n=n, m=["stub_m"], k=["stub_k"])


def compute_all_stub_configs(df: pd.DataFrame) -> dict[str, FeatureConfig]:
    return {col: stub_feature_config(col, df) for col in df.columns}
