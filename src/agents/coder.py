"""
Агент-кодер — реализует признаки и считает статистику.

Три режима работы (вызываются оркестратором):
  - build_features: пишет и запускает код вычисления признаков
  - compute_stats:  считает метрики качества признаков
"""

from __future__ import annotations

import io
import json
import logging
import traceback
from contextlib import redirect_stdout, redirect_stderr
from typing import Any

import numpy as np
import pandas as pd
from langchain_gigachat.chat_models import GigaChat
from scipy.stats import spearmanr
from sklearn.feature_selection import mutual_info_classif
from sklearn.preprocessing import LabelEncoder

logger = logging.getLogger(__name__)

MAX_RETRIES = 3


# ---------------------------------------------------------------------------
# Режим 1: build_features
# ---------------------------------------------------------------------------

BUILD_FEATURES_PROMPT = """Ты — агент-кодер. Напиши Python-код, который вычисляет признаки для train и test.

## Требования к коду

1. Читай данные из папки `data/` (train.csv, test.csv и доп. таблицы).
2. Вычисляй ОДИНАКОВУЮ логику для train и test (без data leakage).
3. Результат сохраняй в переменные `df_train` и `df_test`:
   - `df_train` должен содержать колонки: id_column, target_column, и признаки.
   - `df_test` должен содержать колонки: id_column, и признаки (без target).
4. Обрабатывай пропуски (fillna или drop).
5. Не используй target при вычислении признаков для test.

## Переменные окружения

В коде уже определены:
- `id_col` — имя колонки-идентификатора
- `target_col` — имя колонки-таргета

## Формат ответа

Верни ТОЛЬКО Python-код без объяснений, без markdown-блоков.
Код должен быть самодостаточным (включать все импорты).
"""


def build_features(
    llm: GigaChat,
    analyst_report: dict[str, Any],
    feature_descriptions: list[dict[str, Any]],
) -> tuple[pd.DataFrame | None, pd.DataFrame | None, str]:
    """Генерирует и исполняет код вычисления признаков.

    Returns:
        (df_train, df_test, error_message)
        При успехе error_message == "".
    """
    id_col = analyst_report.get("id_column", "client_id")
    target_col = analyst_report.get("target_column", "target")

    features_json = json.dumps(feature_descriptions, ensure_ascii=False, indent=2)
    report_json = json.dumps(analyst_report, ensure_ascii=False, indent=2)

    user_message = (
        f"## Отчёт аналитика\n```json\n{report_json}\n```\n\n"
        f"## Признаки для реализации\n```json\n{features_json}\n```\n\n"
        f"id_col = '{id_col}'\ntarget_col = '{target_col}'\n\n"
        "Напиши код. Результат — переменные `df_train` и `df_test`."
    )

    messages = [
        {"role": "system", "content": BUILD_FEATURES_PROMPT},
        {"role": "user", "content": user_message},
    ]

    last_error = ""
    for attempt in range(1, MAX_RETRIES + 1):
        logger.info("[coder:build_features] Попытка %d/%d — запрос к GigaChat", attempt, MAX_RETRIES)

        if attempt > 1:
            messages.append({"role": "user", "content": (
                f"Код упал с ошибкой:\n{last_error}\n\n"
                "Исправь код. Верни только исправленный Python-код без объяснений."
            )})

        response = llm.invoke(messages)
        code = _extract_code(response.content)
        logger.info("[coder:build_features] Получен код (%d строк), исполняю...", code.count("\n"))
        messages.append({"role": "assistant", "content": response.content})

        exec_globals: dict[str, Any] = {
            "id_col": id_col,
            "target_col": target_col,
            "__builtins__": __builtins__,
        }

        stdout_buf = io.StringIO()
        stderr_buf = io.StringIO()
        try:
            with redirect_stdout(stdout_buf), redirect_stderr(stderr_buf):
                exec(code, exec_globals)  # noqa: S102
        except Exception:
            last_error = traceback.format_exc()
            logger.warning("[coder:build_features] Ошибка исполнения (попытка %d):\n%s", attempt, last_error)
            continue

        df_train = exec_globals.get("df_train")
        df_test = exec_globals.get("df_test")

        if df_train is None or df_test is None:
            last_error = "Переменные df_train или df_test не определены в коде."
            logger.warning("[coder:build_features] %s", last_error)
            continue

        if not isinstance(df_train, pd.DataFrame) or not isinstance(df_test, pd.DataFrame):
            last_error = "df_train / df_test должны быть pandas.DataFrame."
            logger.warning("[coder:build_features] %s", last_error)
            continue

        feature_cols = [c for c in df_train.columns if c not in (id_col, target_col)]
        logger.info(
            "[coder:build_features] Успех: train=%s, test=%s, признаков=%d: %s",
            df_train.shape, df_test.shape, len(feature_cols), feature_cols,
        )
        return df_train, df_test, ""

    logger.error("[coder:build_features] Все %d попытки завершились ошибкой", MAX_RETRIES)
    return None, None, last_error


# ---------------------------------------------------------------------------
# Режим 2: compute_stats
# ---------------------------------------------------------------------------

def compute_stats(
    df_train: pd.DataFrame,
    target_col: str,
    id_col: str,
) -> dict[str, Any]:
    """Считает метрики качества признаков для критика.

    Возвращает dict со статистиками по каждому признаку:
    pearson, spearman, mutual_info, null_pct, nunique, vif_flag.
    """
    feature_cols = [c for c in df_train.columns if c not in (id_col, target_col)]
    logger.info("[coder:compute_stats] Вычисление метрик для %d признаков: %s", len(feature_cols), feature_cols)

    if not feature_cols:
        logger.warning("[coder:compute_stats] Нет признаков для анализа")
        return {"features": {}}

    y = df_train[target_col].values
    stats: dict[str, dict[str, Any]] = {}

    # Подготавливаем числовую матрицу
    X_encoded = pd.DataFrame(index=df_train.index)
    for col in feature_cols:
        series = df_train[col]
        if pd.api.types.is_numeric_dtype(series):
            X_encoded[col] = series.fillna(-999)
        else:
            le = LabelEncoder()
            X_encoded[col] = le.fit_transform(series.astype(str).fillna("__null__"))

    # Mutual information
    try:
        mi_scores = mutual_info_classif(X_encoded.values, y, random_state=42)
    except Exception as e:
        logger.warning("[coder:compute_stats] mutual_info_classif ошибка: %s", e)
        mi_scores = [0.0] * len(feature_cols)

    for i, col in enumerate(feature_cols):
        series = df_train[col]
        col_enc = X_encoded[col].values

        null_pct = round(float(series.isna().mean()) * 100, 2)
        nunique = int(series.nunique())

        # Pearson
        try:
            pearson = round(float(np.corrcoef(col_enc, y)[0, 1]), 4)
            if np.isnan(pearson):
                pearson = 0.0
        except Exception:
            pearson = 0.0

        # Spearman
        try:
            spearman = round(float(spearmanr(col_enc, y).correlation), 4)
            if np.isnan(spearman):
                spearman = 0.0
        except Exception:
            spearman = 0.0

        mi = round(float(mi_scores[i]), 4)

        stats[col] = {
            "pearson": pearson,
            "spearman": spearman,
            "mutual_info": mi,
            "null_pct": null_pct,
            "nunique": nunique,
        }
        logger.info(
            "[coder:compute_stats]   %s: pearson=%.3f, spearman=%.3f, MI=%.4f, null=%.1f%%",
            col, pearson, spearman, mi, null_pct,
        )

    # VIF (упрощённый: флаг высокой мультиколлинеарности)
    if len(feature_cols) > 1:
        corr_matrix = X_encoded.corr().abs()
        for col in feature_cols:
            other_cols = [c for c in feature_cols if c != col]
            max_corr = corr_matrix.loc[col, other_cols].max()
            stats[col]["max_corr_with_others"] = round(float(max_corr), 4)
            stats[col]["high_collinearity"] = bool(max_corr > 0.85)
            if max_corr > 0.85:
                logger.warning(
                    "[coder:compute_stats]   %s: высокая мультиколлинеарность (%.3f)", col, max_corr
                )
    else:
        for col in feature_cols:
            stats[col]["max_corr_with_others"] = 0.0
            stats[col]["high_collinearity"] = False

    logger.info("[coder:compute_stats] Метрики вычислены для %d признаков", len(stats))
    return {"features": stats}


# ---------------------------------------------------------------------------
# Утилита: извлечение кода из ответа LLM
# ---------------------------------------------------------------------------

def _extract_code(raw: str) -> str:
    """Убирает markdown-обёртку из ответа LLM."""
    text = raw.strip()
    if "```python" in text:
        text = text.split("```python", 1)[1]
        text = text.split("```", 1)[0]
    elif "```" in text:
        text = text.split("```", 1)[1]
        text = text.split("```", 1)[0]
    return text.strip()
