from __future__ import annotations

import logging
import os
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

logger = logging.getLogger(__name__)

CATBOOST_PARAMS = {
    "iterations": 300,
    "learning_rate": 0.05,
    "depth": 6,
    "l2_leaf_reg": 3,
    "random_seed": 42,
    "verbose": 0,
    "thread_count": 1,
    "eval_metric": "AUC",
    "auto_class_weights": "Balanced",
}


@dataclass
class ScoringResult:
    # CV-метрики на train (всегда доступны)
    cv_roc_auc: float = 0.0
    cv_std: float = 0.0
    cv_folds: list = field(default_factory=list)
    # Test-метрики по скрытым меткам (доступны если передан hidden_labels_path)
    test_roc_auc: float | None = None
    test_gini: float | None = None
    # Общая статистика
    n_features: int = 0
    train_rows: int = 0
    test_rows: int = 0
    scoring_elapsed: float = 0.0
    top_features: dict = field(default_factory=dict)


class ScoringEngine:
    def __init__(
        self,
        id_column: str = "client_id",
        target_column: str = "target",
        hidden_labels_path: str | Path | None = None,
        original_columns: set | None = None,
    ):
        self.id_column = id_column
        self.target_column = target_column
        self.hidden_labels_path = Path(hidden_labels_path) if hidden_labels_path else None
        self.original_columns = original_columns

    def score(self, output_dir: str) -> ScoringResult:
        t_start = time.perf_counter()

        train_df = pd.read_csv(os.path.join(output_dir, "train.csv"))
        test_df = pd.read_csv(os.path.join(output_dir, "test.csv"))

        if self.original_columns:
            # Признаки — всё что не в оригинальных колонках input
            feature_cols = [c for c in train_df.columns if c not in self.original_columns]
        else:
            feature_cols = [
                c for c in train_df.columns if c not in (self.id_column, self.target_column)
            ]
        logger.info("[scoring] Признаки (%d): %s", len(feature_cols), feature_cols)

        X_train = train_df[feature_cols].copy()
        y_train = train_df[self.target_column].copy()
        X_test = test_df[feature_cols].copy()

        X_train = X_train.fillna(-999)
        X_test = X_test.fillna(-999)

        cat_features = []
        for i, col in enumerate(X_train.columns):
            if X_train[col].dtype == object or str(X_train[col].dtype) == "str":
                X_train[col] = X_train[col].astype(str)
                X_test[col] = X_test[col].astype(str)
                cat_features.append(i)

        # --- Финальная модель на всём train ---
        logger.info("[scoring] Обучение финальной модели...")
        model = CatBoostClassifier(**CATBOOST_PARAMS, cat_features=cat_features or None)
        model.fit(X_train, y_train)

        # --- 5-fold CV на train ---
        logger.info("[scoring] 5-fold CV...")
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        cv_scores = []
        for fold, (train_idx, val_idx) in enumerate(skf.split(X_train, y_train), 1):
            fold_model = CatBoostClassifier(**CATBOOST_PARAMS, cat_features=cat_features or None)
            fold_model.fit(X_train.iloc[train_idx], y_train.iloc[train_idx])
            probas = fold_model.predict_proba(X_train.iloc[val_idx])[:, 1]
            fold_auc = roc_auc_score(y_train.iloc[val_idx], probas)
            cv_scores.append(fold_auc)
            logger.info("[scoring]   fold %d: AUC=%.4f", fold, fold_auc)
        cv_scores = np.array(cv_scores)

        # --- Test ROC-AUC по скрытым меткам ---
        test_roc_auc = None
        test_gini = None
        if self.hidden_labels_path and self.hidden_labels_path.exists():
            logger.info("[scoring] Загрузка скрытых меток из %s", self.hidden_labels_path)
            labels_df = pd.read_csv(self.hidden_labels_path)
            merged = test_df[[self.id_column]].merge(labels_df, on=self.id_column, how="left")
            hidden_labels = merged[self.target_column].values

            test_probas = model.predict_proba(X_test)[:, 1]
            test_roc_auc = round(float(roc_auc_score(hidden_labels, test_probas)), 6)
            test_gini = round(2 * test_roc_auc - 1, 6)
            logger.info("[scoring] Test ROC-AUC=%.4f, Gini=%.4f", test_roc_auc, test_gini)
        else:
            logger.info("[scoring] Скрытые метки не найдены, test ROC-AUC не вычислен")

        feature_importance = dict(zip(feature_cols, model.get_feature_importance().tolist()))
        top_features = dict(
            sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)[:20]
        )

        scoring_elapsed = time.perf_counter() - t_start

        result = ScoringResult(
            cv_roc_auc=round(float(cv_scores.mean()), 6),
            cv_std=round(float(cv_scores.std()), 6),
            cv_folds=[round(float(s), 6) for s in cv_scores],
            test_roc_auc=test_roc_auc,
            test_gini=test_gini,
            n_features=len(feature_cols),
            train_rows=len(X_train),
            test_rows=len(X_test),
            scoring_elapsed=round(scoring_elapsed, 2),
            top_features=top_features,
        )

        logger.info(
            "[scoring] Готово за %.1f сек. CV AUC=%.4f±%.4f, Test AUC=%s",
            scoring_elapsed, result.cv_roc_auc, result.cv_std,
            f"{test_roc_auc:.4f}" if test_roc_auc is not None else "N/A",
        )
        return result
