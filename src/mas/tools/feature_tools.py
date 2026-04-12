"""Тулы агента признаков: скоринг CatBoost и вспомогательные операции."""
from __future__ import annotations

from typing import Sequence

import pandas as pd
from catboost import CatBoostClassifier, Pool

from src.utils import scoring


def score_features(
    X: pd.DataFrame,
    y: pd.Series,
    feature_names: Sequence[str],
    *,
    random_seed: int = 42,
) -> dict:
    """Оценка полезности набора колонок на train (как в проверке организаторов)."""
    cols = [c for c in feature_names if c in X.columns]
    if not cols:
        return {"metric": float("nan"), "n_features": 0}
    return scoring.train_and_evaluate(X[cols], y, random_seed=random_seed)


def build_pool(
    X: pd.DataFrame,
    y: pd.Series | None,
    feature_names: Sequence[str],
) -> Pool:
    cols = [c for c in feature_names if c in X.columns]
    data = X[cols]
    if y is None:
        return Pool(data=data)
    return Pool(data=data, label=y)


def quick_importance(
    X: pd.DataFrame,
    y: pd.Series,
    feature_names: Sequence[str],
    *,
    iterations: int = 100,
    random_seed: int = 42,
) -> pd.Series:
    """Короткий прогон CatBoost для ранжирования признаков."""
    cols = [c for c in feature_names if c in X.columns]
    if not cols:
        return pd.Series(dtype=float)
    clf = CatBoostClassifier(
        iterations=iterations,
        random_seed=random_seed,
        verbose=False,
        allow_const_label=True,
    )
    clf.fit(X[cols], y)
    imp = clf.get_feature_importance()
    return pd.Series(imp, index=cols).sort_values(ascending=False)
