"""Агент 2: генерация новых фичей через GigaChat.

Поток:
1. Собирает промпт из ctx.eda_report + ctx.readme_text + советов по CatBoost.
2. Вызывает GigaChat-2-Max (reasoning_effort=high).
3. Извлекает Python-код из ответа (блок ```python ... ```).
4. Выполняет код в контролируемом namespace с train_df / test_df.
5. Объединяет лучшие оригинальные фичи + новые -> ctx.feature_matrix_train / _test.
6. При ошибке GigaChat или exec — fallback на базовые числовые трансформации.
"""
from __future__ import annotations

import logging
import re
import unicodedata

import numpy as np
import pandas as pd

from src.mas.context import RunContext
from src.mas.llm import chat_logged, gigachat_client
from src.mas.metrics import compute_metric_m_matrix
from src.mas.prompts import (
    FEATURE_GENERATION_SYSTEM,
    build_feature_generation_prompt,
    build_feature_selection_prompt,
)

logger = logging.getLogger("mas.agent.feature_gen")

_MAX_PAIR_COLS = 6    # лимит для попарных фичей в fallback
_MAX_ORIG_COLS = 15   # максимум оригинальных колонок в итоговой матрице
_CODE_BLOCK_RE = re.compile(r"```python\s*(.*?)```", re.DOTALL | re.IGNORECASE)


def _ascii_col_name(name: str, seen: set) -> str:
    """Приводит имя колонки к безопасному ASCII-идентификатору.

    Шаги:
    1. NFKD-декомпозиция + удаление combining marks (убирает акценты).
    2. Все не-ASCII символы → '_'.
    3. Схлопывание множественных '_', удаление крайних '_'.
    4. Если не начинается с буквы/_ — добавляем 'f_'.
    5. Дедупликация суффиксом _1, _2, ...
    """
    normalized = unicodedata.normalize("NFKD", str(name))
    ascii_only = normalized.encode("ascii", errors="replace").decode("ascii")
    cleaned = re.sub(r"[^A-Za-z0-9_]", "_", ascii_only)
    cleaned = re.sub(r"_+", "_", cleaned).strip("_")
    if not cleaned:
        cleaned = "feat"
    if not (cleaned[0].isalpha() or cleaned[0] == "_"):
        cleaned = "f_" + cleaned
    original = cleaned
    i = 1
    while cleaned in seen:
        cleaned = f"{original}_{i}"
        i += 1
    seen.add(cleaned)
    return cleaned


def _sanitize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Переименовывает все не-ASCII имена колонок в безопасные ASCII-идентификаторы."""
    seen: set = set()
    new_names = [_ascii_col_name(c, seen) for c in df.columns]
    if new_names != list(df.columns):
        mapping = {old: new for old, new in zip(df.columns, new_names) if old != new}
        if mapping:
            logger.info("Переименованы не-ASCII колонки: %s", mapping)
        df = df.rename(columns=dict(zip(df.columns, new_names)))
    return df


def _sanitize_columns_pair(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Переименовывает колонки train/test по одному и тому же mapping.

    Это важно при ASCII-нормализации: если делать rename отдельно,
    одинаковая исходная колонка может получить разные имена из-за коллизий.
    """
    ordered_cols: list[str] = list(train_df.columns) + [
        c for c in test_df.columns if c not in train_df.columns
    ]
    seen: set = set()
    mapping = {col: _ascii_col_name(col, seen) for col in ordered_cols}
    changed = {old: new for old, new in mapping.items() if old != new}
    if changed:
        logger.info("Переименованы не-ASCII/конфликтующие колонки: %s", changed)
    return train_df.rename(columns=mapping), test_df.rename(columns=mapping)


def _sanitize(df: pd.DataFrame) -> pd.DataFrame:
    """Приводит числовые колонки к numpy float64 без inf/NaN/pd.NA.
    Нужно потому что:
    - ratio-фичи дают inf при делении на 0
    - pandas nullable types (Float64, Int64) хранят pd.NA != np.nan,
      и CatBoost / sklearn с ними падают.
    """
    df = df.copy()
    for col in df.columns:
        s = df[col]
        if pd.api.types.is_numeric_dtype(s):
            df[col] = (
                pd.to_numeric(s, errors="coerce")
                .replace([np.inf, -np.inf], np.nan)
                .fillna(0)
                .astype(float)
            )
    return df


def _coerce_y(s: pd.Series) -> pd.Series:
    """Числовой таргет — as-is; строки/категории — Categorical codes (float)."""
    if pd.api.types.is_numeric_dtype(s):
        return s.astype(float)
    return pd.Series(pd.Categorical(s).codes.astype(float), index=s.index)


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

        # Выбираем лучшие оригинальные фичи
        exclude = {ctx.target_col, ctx.id_col}
        orig_cols = self._select_best_orig_cols(ctx, exclude)
        existing = set(orig_cols)

        # Выравниваем индексы — pd.concat требует совпадения
        n_tr = len(ctx.train_frame)
        n_te = len(ctx.test_frame)

        train_orig = ctx.train_frame[orig_cols].reset_index(drop=True)
        test_orig = ctx.test_frame[
            [c for c in orig_cols if c in ctx.test_frame.columns]
        ].reset_index(drop=True)

        new_only_tr = (
            train_new[[c for c in train_new.columns if c not in existing]]
            .reset_index(drop=True)
            .iloc[:n_tr]           # защита от лишних строк из exec
        )
        new_only_te = (
            test_new[[c for c in test_new.columns if c not in existing]]
            .reset_index(drop=True)
            .iloc[:n_te]
        )

        feature_train = _sanitize(
            pd.concat([train_orig, new_only_tr], axis=1).reset_index(drop=True)
        )
        feature_test = _sanitize(
            pd.concat([test_orig, new_only_te], axis=1).reset_index(drop=True)
        )
        ctx.feature_matrix_train, ctx.feature_matrix_test = _sanitize_columns_pair(
            feature_train,
            feature_test,
        )
        # Синхронизируем обе матрицы строго по пересечению колонок.
        # На скрытом датасете новые или исходные признаки могут разъехаться
        # между train/test после exec() или dtype-обработки.
        train_cols = list(ctx.feature_matrix_train.columns)
        test_cols = list(ctx.feature_matrix_test.columns)
        shared = [c for c in train_cols if c in ctx.feature_matrix_test.columns]
        only_train = [c for c in train_cols if c not in ctx.feature_matrix_test.columns]
        only_test = [c for c in test_cols if c not in ctx.feature_matrix_train.columns]

        if only_train:
            logger.warning(
                "Фичи только в train и будут отброшены при выравнивании: %s",
                only_train[:20],
            )
        if only_test:
            logger.warning(
                "Фичи только в test и будут отброшены при выравнивании: %s",
                only_test[:20],
            )

        ctx.feature_matrix_train = ctx.feature_matrix_train[shared].reset_index(drop=True)
        ctx.feature_matrix_test = ctx.feature_matrix_test[shared].reset_index(drop=True)
        ctx.feature_column_names = list(shared)

        logger.info(
            "Итого фичей: %d (%d исходных + %d новых).",
            len(ctx.feature_column_names), len(orig_cols), len(new_only_tr.columns),
        )

    @staticmethod
    def _select_best_orig_cols(ctx: RunContext, exclude: set) -> list[str]:
        """Выбирает не более _MAX_ORIG_COLS оригинальных колонок.

        Числовые — топ по |corr с таргетом|.
        Категориальные (nunique <= 10) — топ по разбросу target rate.
        """
        df = ctx.train_frame
        target = ctx.target_col
        all_cols = [c for c in df.columns if c not in exclude]

        num_cols = [c for c in all_cols if pd.api.types.is_numeric_dtype(df[c])]
        cat_cols = [
            c for c in all_cols
            if not pd.api.types.is_numeric_dtype(df[c])
            and not pd.api.types.is_datetime64_any_dtype(df[c])
            and df[c].nunique() <= 10
        ]

        if target and target in df.columns:
            y_num = _coerce_y(df[target])

            corrs: dict[str, float] = {}
            for c in num_cols:
                try:
                    corrs[c] = abs(float(df[c].corr(y_num)))
                except Exception:
                    corrs[c] = 0.0
            num_sorted = sorted(num_cols, key=lambda c: corrs.get(c, 0.0), reverse=True)

            cat_spread: dict[str, float] = {}
            for c in cat_cols:
                try:
                    tmp = pd.DataFrame({"__col__": df[c], "__y__": y_num}).dropna()
                    grp = tmp.groupby("__col__")["__y__"].mean()
                    cat_spread[c] = float(grp.max() - grp.min())
                except Exception:
                    cat_spread[c] = 0.0
            cat_sorted = sorted(cat_cols, key=lambda c: cat_spread.get(c, 0.0), reverse=True)
        else:
            num_sorted = num_cols
            cat_sorted = cat_cols

        selected = num_sorted[:10] + cat_sorted[:5]
        return selected[:_MAX_ORIG_COLS]

    def _call_gigachat_and_exec(
        self, ctx: RunContext
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Вызывает GigaChat, извлекает код, выполняет exec, возвращает только НОВЫЕ колонки."""
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
        match = _CODE_BLOCK_RE.search(text)
        return match.group(1) if match else text

    def _exec_feature_code(
        self, code: str, ctx: RunContext
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Выполняет код в контролируемом namespace, возвращает только новые колонки."""
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

        # Берём только колонки, которые появились в обоих датафреймах
        shared_new = [c for c in new_cols if c in test_out.columns]
        missing_in_test = set(new_cols) - set(shared_new)
        if missing_in_test:
            logger.warning("Отсутствуют в test после exec: %s", missing_in_test)

        logger.info("exec() успешен, новых колонок: %d", len(shared_new))
        return train_out[shared_new], test_out[shared_new]

    # ------------------------------------------------------------------
    # Fallback — базовые числовые трансформации без GigaChat
    # ------------------------------------------------------------------

    def _fallback_features(
        self, ctx: RunContext
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """log1p, sq, попарные ratio для числовых фичей.
        Оригиналы не дублируем — они добавятся в _generate_feature_matrices.
        """
        logger.info("Fallback: генерируем базовые числовые трансформации.")
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
            e = (
                ctx.test_frame[col].fillna(med)
                if col in ctx.test_frame.columns
                else pd.Series([med] * len(ctx.test_frame), index=ctx.test_frame.index)
            )
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
                denom_e = (
                    ctx.test_frame[cj].fillna(med_j).replace(0, np.nan)
                    if cj in ctx.test_frame.columns
                    else pd.Series([np.nan] * len(ctx.test_frame), index=ctx.test_frame.index)
                )
                feats_tr[f"ratio_{ci}_div_{cj}"] = (
                    ctx.train_frame[ci].fillna(med_i) / denom_t
                ).fillna(0)
                feats_te[f"ratio_{ci}_div_{cj}"] = (
                    (
                        ctx.test_frame[ci].fillna(med_i)
                        if ci in ctx.test_frame.columns
                        else pd.Series([med_i] * len(ctx.test_frame), index=ctx.test_frame.index)
                    )
                    / denom_e
                ).fillna(0)

        return (
            pd.DataFrame(feats_tr, index=ctx.train_frame.index),
            pd.DataFrame(feats_te, index=ctx.test_frame.index),
        )

    # ------------------------------------------------------------------
    # Шаги 2-3: метрики и промпт
    # ------------------------------------------------------------------

    def _compute_metrics(self, ctx: RunContext) -> None:
        names = list(ctx.feature_column_names)
        if not names or ctx.feature_matrix_train is None or ctx.feature_matrix_train.empty:
            ctx.metric_m_matrix = np.zeros((0, 0), dtype=float)
            return

        X = ctx.feature_matrix_train
        ctx.metric_m_matrix = compute_metric_m_matrix(X, names, context=None)

    def _refresh_selection_prompt(self, ctx: RunContext) -> None:
        excerpt = (ctx.readme_text or "")[:4000]
        ctx.selection_prompt = build_feature_selection_prompt(
            readme_excerpt=excerpt,
            feature_names=list(ctx.feature_column_names),
            metric_m_matrix=ctx.metric_m_matrix if ctx.metric_m_matrix is not None else np.zeros((0, 0)),
            extra_hints=ctx.schema_notes,
        )
