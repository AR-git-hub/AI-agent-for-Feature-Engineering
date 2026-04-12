"""Метрики для агента 2: по одной фиче (n) и по парам (m). Здесь — заглушки, подмените реализацией."""
from __future__ import annotations

import numpy as np
import pandas as pd


def metric_n(feature: pd.Series, context: dict | None = None) -> float:
    """
    Заглушка «метрика n» для одной фичи.
    Должна вернуть одно число (например, важность, корреляция с таргетом, стабильность).
    context — опционально: таргет, веса, метаданные; заполните при реализации.
    """
    _ = feature, context
    return 0.0


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


def compute_metric_n_vector(
    X: pd.DataFrame,
    feature_cols: list[str],
    *,
    context: dict | None = None,
) -> np.ndarray:
    """
    П.1 ТЗ агента 2: применить metric_n к каждой фиче, результат — вектор длины len(feature_cols).
    """
    out = np.zeros(len(feature_cols), dtype=float)
    for idx, name in enumerate(feature_cols):
        if name not in X.columns:
            out[idx] = float("nan")
            continue
        out[idx] = metric_n(X[name], context)
    return out


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
