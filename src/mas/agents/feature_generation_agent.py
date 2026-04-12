"""Агент 2: генерация новых фичей через GigaChat.

Поток:
1. Собирает промпт из ctx.eda_report + ctx.readme_text + советов по CatBoost.
2. Вызывает GigaChat-2-Max (reasoning_effort=high).
3. Извлекает Python-код из ответа (блок ```python ... ```).
4. Выполняет код в контролируемом namespace с train_df / test_df.
5. Собирает новые колонки → ctx.feature_matrix_train / _test.
6. При ошибке GigaChat или exec — fallback на базовые числовые трансформации.
"""
from __future__ import annotations

import logging
import re

import numpy as np
import pandas as pd

from src.mas.context import RunContext
from src.mas.llm import chat_logged, gigachat_client
from src.mas.metrics import compute_metric_m_matrix, compute_metric_n_vector
from src.mas.prompts import (
    FEATURE_GENERATION_SYSTEM,
    build_feature_generation_prompt,
    build_feature_selection_prompt,
)

logger = logging.getLogger("mas.agent.feature_gen")

_MAX_PAIR_COLS = 6          # Лимит для попарных фичей в fallback
_CODE_BLOCK_RE = re.compile(r"```python\s*(.*?)```", re.DOTALL | re.IGNORECASE)


class FeatureGenerationAgent:
    def run(self, ctx: RunContext) -> RunContext:
        self._generate_feature_matrices(ctx)
        self._compute_metrics(ctx)
        self._refresh_selection_prompt(ctx)
        return ctx

    # ------------------------------------------------------------------
    # Шаг 1 — генерация
    # ------------------------------------------------------------------

    def _generate_feature_matrices(self, ctx: RunContext) -> None:
        if ctx.train_frame is None or ctx.test_frame is None:
            logger.warning("train_frame/test_frame пусты — пропускаем генерацию.")
            ctx.feature_matrix_train = pd.DataFrame()
            ctx.feature_matrix_test = pd.DataFrame()
            ctx.feature_column_names = []
            return

        try:
            train_new, test_new = self._call_gigachat_and_exec(ctx)
        except Exception as exc:
            logger.warning("GigaChat-генерация упала (%s) — переходим на fallback.", exc)
            train_new, test_new = self._fallback_features(ctx)

        ctx.feature_matrix_train = train_new.reset_index(drop=True)
        ctx.feature_matrix_test = test_new.reset_index(drop=True)
        ctx.feature_column_names = list(train_new.columns)
        logger.info("Итого фичей: %d", len(ctx.feature_column_names))

    def _call_gigachat_and_exec(
        self, ctx: RunContext
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Вызывает GigaChat, извлекает код, выполняет exec, возвращает новые колонки."""
        from gigachat.models import Chat, Messages, MessagesRole

        exclude = {ctx.target_col, ctx.id_col}
        available_cols = [c for c in ctx.train_frame.columns if c not in exclude]

        user_text = build_feature_generation_prompt(
            eda_report=ctx.eda_report,
            readme=ctx.readme_text,
            available_columns=available_cols,
        )

        payload = Chat(
            messages=[
                Messages(role=MessagesRole.SYSTEM, content=FEATURE_GENERATION_SYSTEM),
                Messages(role=MessagesRole.USER, content=user_text),
            ],
            reasoning_effort="high",
        )

        with gigachat_client() as client:
            completion = chat_logged(client, payload, label="FeatureGen")

        raw_content = completion.choices[0].message.content or ""
        code = self._extract_code(raw_content)

        if not code.strip():
            raise ValueError("GigaChat не вернул Python-блок с кодом.")

        logger.info("Извлечён код (%d символов), выполняем exec().", len(code))
        return self._exec_feature_code(code, ctx)

    @staticmethod
    def _extract_code(text: str) -> str:
        """Вытаскивает содержимое первого ```python ... ``` блока."""
        match = _CODE_BLOCK_RE.search(text)
        if match:
            return match.group(1)
        # Если модель не обернула в блок — возвращаем весь текст как есть.
        return text

    def _exec_feature_code(
        self, code: str, ctx: RunContext
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """
        Выполняет код в контролируемом namespace.
        Namespace содержит только: train_df, test_df, pd, np.
        Возвращает только НОВЫЕ колонки (не из исходного train_frame).
        """
        original_cols = set(ctx.train_frame.columns)

        namespace: dict = {
            "train_df": ctx.train_frame.copy(),
            "test_df": ctx.test_frame.copy(),
            "pd": pd,
            "np": np,
        }
        try:
            exec(code, namespace)  # noqa: S102
        except Exception as exc:
            logger.error("exec() упал: %s\nКод:\n%s", exc, code[:800])
            raise

        train_out: pd.DataFrame = namespace["train_df"]
        test_out: pd.DataFrame = namespace["test_df"]

        new_cols = [c for c in train_out.columns if c not in original_cols]
        if not new_cols:
            raise ValueError("После exec() новых колонок не появилось.")

        # Убедимся, что те же колонки есть и в test
        test_new_cols = [c for c in new_cols if c in test_out.columns]
        missing_in_test = set(new_cols) - set(test_new_cols)
        if missing_in_test:
            logger.warning("Отсутствуют в test после exec: %s", missing_in_test)

        logger.info("exec() успешен, новых колонок: %d", len(test_new_cols))
        return train_out[test_new_cols], test_out[test_new_cols]

    # ------------------------------------------------------------------
    # Fallback — базовые числовые трансформации без GigaChat
    # ------------------------------------------------------------------

    def _fallback_features(
        self, ctx: RunContext
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Базовые трансформации: log1p, sq, попарные ratio."""
        logger.info("Fallback: генерируем базовые числовые фичи.")
        exclude = {ctx.target_col, ctx.id_col}
        num_cols = [
            c for c in ctx.train_frame.select_dtypes(include="number").columns
            if c not in exclude
        ]

        feats_tr: dict[str, pd.Series] = {}
        feats_te: dict[str, pd.Series] = {}

        for col in num_cols:
            med = float(ctx.train_frame[col].median())
            t = ctx.train_frame[col].fillna(med)
            e = ctx.test_frame[col].fillna(med) if col in ctx.test_frame.columns else pd.Series(
                [med] * len(ctx.test_frame), index=ctx.test_frame.index
            )
            feats_tr[col] = t
            feats_te[col] = e
            if t.min() >= 0:
                feats_tr[f"log1p_{col}"] = np.log1p(t)
                feats_te[f"log1p_{col}"] = np.log1p(e)
            feats_tr[f"sq_{col}"] = t ** 2
            feats_te[f"sq_{col}"] = e ** 2

        for i, ci in enumerate(num_cols[:_MAX_PAIR_COLS]):
            for cj in num_cols[i + 1:_MAX_PAIR_COLS]:
                med_j = float(ctx.train_frame[cj].median())
                med_i = float(ctx.train_frame[ci].median())
                denom_t = ctx.train_frame[cj].fillna(med_j).replace(0, np.nan)
                denom_e = ctx.test_frame[cj].fillna(med_j).replace(0, np.nan) if cj in ctx.test_frame.columns else pd.Series(
                    [np.nan] * len(ctx.test_frame), index=ctx.test_frame.index
                )
                feats_tr[f"ratio_{ci}_div_{cj}"] = (ctx.train_frame[ci].fillna(med_i) / denom_t).fillna(0)
                feats_te[f"ratio_{ci}_div_{cj}"] = (
                    (ctx.test_frame[ci].fillna(med_i) if ci in ctx.test_frame.columns
                     else pd.Series([med_i] * len(ctx.test_frame), index=ctx.test_frame.index))
                    / denom_e
                ).fillna(0)

        idx_tr = ctx.train_frame.index
        idx_te = ctx.test_frame.index
        return (
            pd.DataFrame(feats_tr, index=idx_tr),
            pd.DataFrame(feats_te, index=idx_te),
        )

    # ------------------------------------------------------------------
    # Шаги 2-3: метрики и промпт агента отбора
    # ------------------------------------------------------------------

    def _compute_metrics(self, ctx: RunContext) -> None:
        names = list(ctx.feature_column_names)
        if not names or ctx.feature_matrix_train is None or ctx.feature_matrix_train.empty:
            ctx.metric_n_vector = np.array([], dtype=float)
            ctx.metric_m_matrix = np.zeros((0, 0), dtype=float)
            return
        X = ctx.feature_matrix_train
        ctx.metric_n_vector = compute_metric_n_vector(X, names, context=None)
        ctx.metric_m_matrix = compute_metric_m_matrix(X, names, context=None)

    def _refresh_selection_prompt(self, ctx: RunContext) -> None:
        excerpt = (ctx.readme_text or "")[:4000]
        ctx.selection_prompt = build_feature_selection_prompt(
            readme_excerpt=excerpt,
            feature_names=list(ctx.feature_column_names),
            metric_n_vector=ctx.metric_n_vector if ctx.metric_n_vector is not None else np.array([]),
            metric_m_matrix=ctx.metric_m_matrix if ctx.metric_m_matrix is not None else np.zeros((0, 0)),
            extra_hints=ctx.schema_notes,
        )
