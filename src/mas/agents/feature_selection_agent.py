"""Агент 3: отбор фичей через RFE.

Поток:
1. Берём feature_matrix_train и y из train_frame.
2. Запускаем select_features_rfe (RandomForest + sklearn RFE).
3. Строим EDA-отчёт по сгенерированным фичам с RFE-рангом → ctx.features_eda_report.
4. Сохраняем выбранные имена в ctx.selected_feature_names (≤ MAX_FEATURES).
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from src.mas.context import RunContext
from src.mas.tools.rfe_tools import select_features_rfe

logger = logging.getLogger("mas.agent.feature_sel")

MAX_FEATURES = 5


class FeatureSelectionAgent:
    def run(self, ctx: RunContext) -> RunContext:
        if ctx.feature_matrix_train is None or ctx.feature_matrix_train.empty:
            logger.warning("feature_matrix_train пуст — отбор пропущен.")
            ctx.selection_notes = "Нечего отбирать: матрица фичей пуста."
            ctx.selected_feature_names = []
            return ctx

        if ctx.target_col is None or ctx.train_frame is None:
            logger.warning("target_col или train_frame не заданы — отбор пропущен.")
            ctx.selection_notes = "Нечего отбирать: нет таргета."
            ctx.selected_feature_names = []
            return ctx

        y = ctx.train_frame[ctx.target_col]
        valid_cols = [c for c in ctx.feature_column_names if c in ctx.feature_matrix_train.columns]

        if not valid_cols:
            logger.warning("Нет валидных колонок в feature_matrix_train — отбор пропущен.")
            ctx.selection_notes = "Нечего отбирать: нет валидных колонок."
            ctx.selected_feature_names = []
            return ctx

        X = ctx.feature_matrix_train[valid_cols]

        n_select = min(MAX_FEATURES, len(valid_cols))
        logger.info("Запускаем RFE: %d фичей → выбираем %d.", len(valid_cols), n_select)

        selected, ranking, _ = select_features_rfe(X, y, n_features_to_select=n_select)

        ctx.features_eda_report = self._build_features_eda_report(X, y, ranking, valid_cols)

        ctx.selected_feature_names = selected[:MAX_FEATURES]
        ctx.selection_notes = (
            f"RFE выбрал {len(ctx.selected_feature_names)} фичей: {ctx.selected_feature_names}"
        )
        logger.info(ctx.selection_notes)
        return ctx

    # ------------------------------------------------------------------
    # EDA-отчёт по сгенерированным фичам
    # ------------------------------------------------------------------

    @staticmethod
    def _build_features_eda_report(
        X: pd.DataFrame,
        y: pd.Series,
        ranking: np.ndarray,
        col_names: list[str],
    ) -> str:
        """
        Строка на фичу:
          feat  null=0%  mean=…  std=…  skew=…  corr_target=+0.12  rfe_rank=1
        Отсортировано по rfe_rank (1 = лучший по RFE).
        """
        rank_map = dict(zip(col_names, ranking))
        lines = [f"=== Features EDA (RFE): {X.shape[0]} строк, {X.shape[1]} фичей ==="]
        cols_sorted = sorted(col_names, key=lambda c: rank_map.get(c, 9999))
        for col in cols_sorted:
            s = X[col]
            valid = s.dropna()
            corr = ""
            if len(valid) > 1:
                try:
                    corr = f" corr_target={valid.corr(y.loc[valid.index]):+.3f}"
                except Exception:
                    pass
            rk = rank_map.get(col, "?")
            selected_mark = " ★" if rk == 1 else ""
            lines.append(
                f"  {col}  null={s.isna().mean():.1%}"
                f"  mean={valid.mean():.4g}  std={valid.std():.4g}"
                f"  skew={valid.skew():.2f}{corr}  rfe_rank={rk}{selected_mark}"
            )
        return "\n".join(lines)
