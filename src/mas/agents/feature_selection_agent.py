"""Агент 3: отбор признаков для CatBoost через GigaChat.

Поток:
1. Быстрый CatBoost (150 итераций) -> importance каждой фичи.
2. EDA-отчёт: null%, mean, std, skew, corr_target, importance.
3. GigaChat читает отчёт + readme -> рассуждает -> возвращает JSON <=5 имён.
4. Fallback: топ-5 по CatBoost importance если GigaChat упал или не распарсился.
"""
from __future__ import annotations

import json
import logging
import re

import pandas as pd
from catboost import CatBoostClassifier
from gigachat.models import Chat, Messages, MessagesRole

from src.mas.context import RunContext
from src.mas.llm import chat_logged, gigachat_client
from src.mas.prompts import FEATURE_SELECTION_SYSTEM, build_feature_selection_llm_prompt

logger = logging.getLogger("mas.agent.feature_sel")

MAX_FEATURES = 5
_IMPORTANCE_ITERATIONS = 150
_JSON_RE = re.compile(r"```json\s*(.*?)```", re.DOTALL | re.IGNORECASE)


def _coerce_y(s: pd.Series) -> pd.Series:
    """Числовой таргет — as-is; строки/категории — Categorical codes (float)."""
    if pd.api.types.is_numeric_dtype(s):
        return s.astype(float)
    return pd.Series(pd.Categorical(s).codes.astype(float), index=s.index)


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

        valid_cols = [
            c for c in ctx.feature_column_names
            if c in ctx.feature_matrix_train.columns
        ]
        if not valid_cols:
            logger.warning("Нет валидных колонок в feature_matrix_train — отбор пропущен.")
            ctx.selection_notes = "Нечего отбирать: нет валидных колонок."
            ctx.selected_feature_names = []
            return ctx

        # Предварительный отбор по n_unique: выкидываем константные признаки.
        valid_cols = [
            c for c in valid_cols
            if ctx.feature_n_unique.get(c, 0) > 1
        ]
        if not valid_cols:
            logger.warning("После фильтра по n_unique>1 валидных колонок не осталось.")
            ctx.selection_notes = "Нечего отбирать: все признаки константные по n_unique."
            ctx.selected_feature_names = []
            return ctx

        X = ctx.feature_matrix_train[valid_cols].reset_index(drop=True)
        y = _coerce_y(ctx.train_frame[ctx.target_col]).reset_index(drop=True)

        # Шаг 1 — CatBoost importance
        importance = self._catboost_importance(X, y)
        logger.info("CatBoost importance рассчитан по %d фичам.", len(importance))

        # Шаг 2 — EDA-отчёт с importance -> ctx.features_eda_report
        ctx.features_eda_report = self._build_features_eda_report(
            X, y, importance, ctx.feature_n_unique
        )

        # Шаг 3 — GigaChat выбирает фичи
        selected = self._select_with_gigachat(ctx, valid_cols)

        # Шаг 4 — fallback
        if not selected:
            logger.warning(
                "GigaChat не выбрал фичи — fallback: отбор по n_unique, затем importance, затем metric_n."
            )
            selected = self._fallback_select(ctx, importance, valid_cols)

        ctx.selected_feature_names = selected[:MAX_FEATURES]
        ctx.selection_notes = (
            f"Выбрано {len(ctx.selected_feature_names)}: {ctx.selected_feature_names}"
        )
        logger.info(ctx.selection_notes)
        return ctx

    # ------------------------------------------------------------------
    # Шаг 1 — CatBoost importance
    # ------------------------------------------------------------------

    @staticmethod
    def _catboost_importance(X: pd.DataFrame, y: pd.Series) -> pd.Series:
        """Считает CatBoost feature importance.

        Всё кодируется в числа — никаких строк в CatBoost.
        Это единственный способ избежать UnicodeDecodeError в C++-слое CatBoost,
        когда категориальные значения содержат кириллицу или другие не-ASCII символы.

        Кодирование:
        - datetime       -> int64 (наносекунды)
        - numeric        -> float64, inf/NaN/pd.NA -> 0
        - всё остальное  -> Categorical.codes (целые числа), пропуски -> -1
        """
        import numpy as np

        orig_cols = list(X.columns)
        # Используем безопасные ASCII-имена f_0, f_1, ... чтобы гарантированно
        # избежать UnicodeDecodeError в C++-слое CatBoost при любых именах колонок.
        safe_cols = [f"f_{i}" for i in range(len(orig_cols))]
        col_map = dict(zip(safe_cols, orig_cols))

        prepared: dict = {}
        for safe, col in zip(safe_cols, orig_cols):
            s = X[col]
            if pd.api.types.is_datetime64_any_dtype(s):
                prepared[safe] = pd.to_numeric(s, errors="coerce").fillna(0).astype(float)
            elif pd.api.types.is_numeric_dtype(s):
                prepared[safe] = (
                    pd.to_numeric(s, errors="coerce")
                    .replace([np.inf, -np.inf], np.nan)
                    .fillna(0)
                    .astype(float)
                )
            else:
                codes = pd.Categorical(s.astype(object).fillna(np.nan)).codes.astype(float)
                prepared[safe] = codes

        X_ready = pd.DataFrame(prepared, index=X.index)

        # train_dir изолирован в output/.cb_imp_tmp — чтобы catboost_info в корне
        # не засорялась бинарными tfevents-файлами, которые ломают scoring.py на fold 3.
        clf = CatBoostClassifier(
            iterations=_IMPORTANCE_ITERATIONS,
            random_seed=42,
            verbose=False,
            allow_const_label=True,
            train_dir="output/.cb_imp_tmp",
        )
        clf.fit(X_ready, y)
        imp_safe = pd.Series(clf.get_feature_importance(), index=safe_cols)
        # Возвращаем с оригинальными именами колонок
        return imp_safe.rename(index=col_map).sort_values(ascending=False)

    # ------------------------------------------------------------------
    # Шаг 2 — EDA-отчёт по сгенерированным фичам
    # ------------------------------------------------------------------

    @staticmethod
    def _build_features_eda_report(
        X: pd.DataFrame,
        y: pd.Series,
        importance: pd.Series,
        feature_n_unique: dict[str, int] | None = None,
    ) -> str:
        """Строка на фичу: null%, mean/std/skew (числовые), corr_target, importance.
        Сортировка по убыванию importance.
        """
        lines = [f"=== Features EDA: {X.shape[0]} строк, {X.shape[1]} фичей ==="]
        for col in importance.index:
            if col not in X.columns:
                continue
            s = X[col]
            imp = importance.get(col, 0.0)
            corr_str = ""
            try:
                valid = s.dropna()
                if pd.api.types.is_numeric_dtype(s) and len(valid) > 1:
                    corr_val = float(valid.corr(y.loc[valid.index]))
                    corr_str = f" corr_target={corr_val:+.3f}"
            except Exception:
                pass

            if pd.api.types.is_numeric_dtype(s):
                valid = s.dropna()
                lines.append(
                    f"  {col}  null={s.isna().mean():.1%}"
                    f"  n_unique={feature_n_unique.get(col, s.nunique(dropna=False)) if feature_n_unique else s.nunique(dropna=False)}"
                    f"  mean={valid.mean():.4g}  std={valid.std():.4g}"
                    f"  skew={valid.skew():.2f}{corr_str}  importance={imp:.2f}"
                )
            else:
                lines.append(
                    f"  {col}  [categ]  null={s.isna().mean():.1%}"
                    f"  n_unique={feature_n_unique.get(col, s.nunique(dropna=False)) if feature_n_unique else s.nunique(dropna=False)}"
                    f"{corr_str}  importance={imp:.2f}"
                )
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Шаг 3 — GigaChat отбирает фичи
    # ------------------------------------------------------------------

    def _select_with_gigachat(self, ctx: RunContext, valid_cols: list[str]) -> list[str]:
        user_text = build_feature_selection_llm_prompt(
            features_eda_report=ctx.features_eda_report,
            readme=ctx.readme_text,
            available_features=valid_cols,
            original_eda_report=ctx.eda_report,
            metric_n_results=ctx.metric_n_results or None,
            feature_n_unique=ctx.feature_n_unique or None,
        )
        payload = Chat(
            messages=[
                Messages(role=MessagesRole.SYSTEM, content=FEATURE_SELECTION_SYSTEM),
                Messages(role=MessagesRole.USER, content=user_text),
            ],
            reasoning_effort="high",
        )
        try:
            with gigachat_client() as client:
                completion = chat_logged(client, payload, label="Agent3-Selection")
            content = completion.choices[0].message.content or ""
            return self._parse_selected_features(content, valid_cols)
        except Exception as exc:
            logger.warning("GigaChat упал при отборе фичей: %s", exc)
            return []

    @staticmethod
    def _parse_selected_features(content: str, valid_cols: list[str]) -> list[str]:
        valid_set = set(valid_cols)

        match = _JSON_RE.search(content)
        raw = match.group(1) if match else content.strip()
        try:
            parsed = json.loads(raw)
            if isinstance(parsed, list):
                names = [str(n) for n in parsed if str(n) in valid_set]
                if names:
                    logger.info("GigaChat выбрал фичи: %s", names)
                    return names[:MAX_FEATURES]
        except json.JSONDecodeError:
            pass

        # Fallback: ищем имена прямо в тексте ответа
        found = [c for c in valid_cols if c in content]
        if found:
            logger.info("Фичи найдены в тексте ответа: %s", found[:MAX_FEATURES])
            return found[:MAX_FEATURES]

        logger.warning("Не удалось распарсить фичи из ответа GigaChat:\n%s", content[:300])
        return []

    @staticmethod
    def _fallback_select(
        ctx: RunContext,
        importance: pd.Series,
        valid_cols: list[str],
    ) -> list[str]:
        """Жёсткий fallback: сначала n_unique, потом importance, потом metric_n."""
        ranked = sorted(
            valid_cols,
            key=lambda col: (
                -ctx.feature_n_unique.get(col, -1),
                -float(importance.get(col, 0.0)),
                float(ctx.metric_n_results.get(col, float("inf"))),
                col,
            ),
        )
        return ranked[:MAX_FEATURES]
