"""Метрики для агента 2: по парам фичей (m)."""
from __future__ import annotations

import numpy as np
import pandas as pd


def metric_m(
    feature_i: pd.Series,
    feature_j: pd.Series,
    context: dict | None = None,
) -> float:
    """
    Заглушка «метрика m» для пары фичей.
    Пример смысла: взаимная информация, корреляция, interaction gain — на ваш выбор.
    """
    _ = feature_i, feature_j, context
    return 0.0


def compute_metric_m_matrix(
    X: pd.DataFrame,
    feature_cols: list[str],
    *,
    context: dict | None = None,
) -> np.ndarray:
    """
    П.2 ТЗ агента 2: применить metric_m ко всем парам (i, j), матрица n×n.
    """
    n = len(feature_cols)
    mat = np.zeros((n, n), dtype=float)
    for i, ni in enumerate(feature_cols):
        for j, nj in enumerate(feature_cols):
            if ni not in X.columns or nj not in X.columns:
                mat[i, j] = float("nan")
                continue
            mat[i, j] = metric_m(X[ni], X[nj], context)
    return mat
