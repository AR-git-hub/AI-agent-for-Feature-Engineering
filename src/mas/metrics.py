"""Метрики для агентов: metric_n (примесь по 2-уровневому дереву) и metric_m (заглушка).

metric_n реализована через алгоритм двухуровневого бинарного дерева:
  - getBest  — находит лучший порог разбиения одного массива пар (value, target)
  - countMetric — два этажа: корень → два дочерних узла → возвращает одно число m2+m3
Чем НИЖЕ metric_n, тем лучше признак разделяет классы (как entropy/gini в дереве решений).
"""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

METRIC_N_STEP = 2000


# ─────────────────────────────────────────────────────────────────────────────
# Низкоуровневые функции (предоставлены пользователем)
# ─────────────────────────────────────────────────────────────────────────────

def calc_impurity(arr: list[tuple[Any, int]]) -> float:
    """Примесь одного листа: ошибка мажоритарного классификатора."""
    if not arr:
        return 0
    targets = [x[1] for x in arr]
    pred = 0 if sum(targets) / len(targets) < 0.5 else 1
    return float(sum(abs(pred - y) for y in targets))


def getBest(
    arr: list[tuple[Any, int]],
    step: int,
) -> tuple[Any, list, list, float]:
    """Найти лучший порог разбиения массива пар (value, target).

    Возвращает (threshold, left, right, impurity).
    Если разбиение невозможно — (None, arr, [], impurity(arr)).
    """
    if not arr or len(arr) <= 1:
        return None, arr, [], calc_impurity(arr)

    best: list = [None, [], [], float("inf")]

    sorted_x = sorted(set(x[0] for x in arr))

    for i in range(1, len(sorted_x), step):
        f = sorted_x[i]
        l = [x for x in arr if x[0] < f]
        r = [x for x in arr if x[0] >= f]

        if not l or not r:
            continue

        imp = calc_impurity(l) + calc_impurity(r)
        if imp < best[3]:
            best = [f, l, r, imp]

    if best[3] == float("inf"):
        return None, arr, [], calc_impurity(arr)
    return best[0], best[1], best[2], best[3]


def countMetric(
    arr: list[tuple[Any, int]],
    step: int,
) -> float:
    """Двухуровневое дерево → суммарная примесь m2+m3 (одно число).

    Этаж 1: корневое разбиение → l1, r1
    Этаж 2: l1 → m2, r1 → m3
    Возвращает m2 + m3.
    """
    # Этаж 1
    _, l1, r1, _ = getBest(arr, step)
    if not l1 and not r1:
        l1, r1 = arr, []

    # Этаж 2
    _, _, _, m2 = getBest(l1, step)
    _, _, _, m3 = getBest(r1, step)

    return m2 + m3


# ─────────────────────────────────────────────────────────────────────────────
# Публичный API
# ─────────────────────────────────────────────────────────────────────────────

def metric_n(feature: pd.Series, context: dict | None = None) -> float:
    """Метрика N для одного признака: суммарная примесь двухуровневого дерева.

    Чем ниже — тем лучше признак разделяет целевую переменную.

    context должен содержать ключ "target" (pd.Series | list) для корректного расчёта.
    Без target возвращает 0.0.
    """
    target = (context or {}).get("target")
    if target is None or len(feature) == 0:
        return 0.0

    feat_num = pd.to_numeric(feature, errors="coerce").fillna(0).tolist()
    tgt_num = pd.to_numeric(
        pd.Series(target).reset_index(drop=True), errors="coerce"
    ).fillna(0).round().astype(int).tolist()

    if len(feat_num) != len(tgt_num):
        n = min(len(feat_num), len(tgt_num))
        feat_num, tgt_num = feat_num[:n], tgt_num[:n]

    # Пользовательский эталон: считаем строго как countMetric(arr, 30).
    arr = [[x, y] for x, y in zip(feat_num, tgt_num)]
    return float(countMetric(arr, METRIC_N_STEP))


def compute_metric_n_vector(
    X: pd.DataFrame,
    feature_cols: list[str],
    *,
    context: dict | None = None,
) -> np.ndarray:
    """Вектор metric_n по каждой фиче из feature_cols."""
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
    """Матрица n×n заглушки metric_m (не используется в текущем пайплайне)."""
    n = len(feature_cols)
    return np.zeros((n, n), dtype=float)
