"""RFE-отбор признаков на базе RandomForestClassifier."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import RFE


def select_features_rfe(
    X: pd.DataFrame | np.ndarray,
    y: pd.Series | np.ndarray,
    n_features_to_select: int = 5,
    random_state: int = 42,
    n_jobs: int = 1,
) -> tuple[list, np.ndarray, np.ndarray]:
    """
    Отбор признаков с помощью RFE + RandomForest.

    Параметры
    ----------
    X : DataFrame или ndarray — матрица признаков
    y : Series или ndarray  — целевая переменная
    n_features_to_select : сколько оставить
    random_state, n_jobs   : проброс в RandomForestClassifier

    Возвращает
    ----------
    selected_features : list — имена (если DataFrame) или индексы выбранных признаков
    ranking           : ndarray — ранг каждого признака (1 = лучший)
    support_mask      : ndarray bool — маска выбранных признаков
    """
    feature_names: list | None = None
    if hasattr(X, "columns"):
        feature_names = list(X.columns)
        X_values = X.values
    else:
        X_values = np.asarray(X)

    model = RandomForestClassifier(
        n_estimators=100,
        random_state=random_state,
        n_jobs=n_jobs,
    )
    rfe = RFE(estimator=model, n_features_to_select=n_features_to_select, step=1)
    rfe.fit(X_values, y)

    support_mask: np.ndarray = rfe.support_
    ranking: np.ndarray = rfe.ranking_

    if feature_names is not None:
        selected_features = [name for name, flag in zip(feature_names, support_mask) if flag]
    else:
        selected_features = list(np.where(support_mask)[0])

    return selected_features, ranking, support_mask
