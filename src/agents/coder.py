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

from src.utils.retry import with_retry
from sklearn.feature_selection import mutual_info_classif
from sklearn.preprocessing import LabelEncoder

logger = logging.getLogger(__name__)

MAX_RETRIES = 3


# ---------------------------------------------------------------------------
# Режим 1: build_features
# ---------------------------------------------------------------------------

BUILD_FEATURES_PROMPT = """
Ты — агент-кодер. Пишешь Python-код для построения признаков.
НЕ придумываешь признаки, НЕ оцениваешь качество. Только реализуешь код.

## Правила написания кода

- Только pandas и numpy. Не pip install, не импортируй ничего кроме них.
- Верни ТОЛЬКО Python-код. Без markdown, без пояснений, без комментариев.

---

## ОБЯЗАТЕЛЬНЫЙ ШАБЛОН КОДА (строго следуй ему)

Все переменные уже существуют в памяти. НЕ читай файлы через pd.read_csv.
Используй переменные напрямую: train, test, и вспомогательные таблицы.

```python
import numpy as np
import pandas as pd

original_train_len = len(train)
original_test_len = len(test)

# Шаг 1: Подготовь агрегаты из вспомогательных таблиц (ТОЛЬКО по train)
# Пример: агрегация из users_df по user_id
# user_agg = users_df.groupby('user_id').agg(...).reset_index()

# Шаг 2: Замапь признаки в train и test через .map() или .merge()
# ВАЖНО: мёрдж делай НЕ по id_col, а по user_id или product_id
# train_work = train.copy()
# test_work = test.copy()
# train_work['feat'] = train_work['user_id'].map(user_agg.set_index('user_id')['feat'])
# test_work['feat'] = test_work['user_id'].map(user_agg.set_index('user_id')['feat'])

# Шаг 3: Заполни NaN и inf медианой по train
feature_cols = ['feat_1', 'feat_2', ...]  # замени на реальные имена
for col in feature_cols:
    train_work[col] = train_work[col].replace([np.inf, -np.inf], np.nan)
    test_work[col] = test_work[col].replace([np.inf, -np.inf], np.nan)
    med = train_work[col].median()
    fill = med if not pd.isna(med) else 0
    train_work[col] = train_work[col].fillna(fill)
    test_work[col] = test_work[col].fillna(fill)

# Шаг 4: Собери итоговые датафреймы
df_train = train_work[[id_col, target_col] + feature_cols].copy()
df_test = test_work[[id_col] + feature_cols].copy()

assert len(df_train) == original_train_len, f"train изменил размер: {len(df_train)} != {original_train_len}"
assert len(df_test) == original_test_len, f"test изменил размер: {len(df_test)} != {original_test_len}"
assert df_train[feature_cols].isna().sum().sum() == 0, "NaN в df_train"
assert df_test[feature_cols].isna().sum().sum() == 0, "NaN в df_test"
print("BUILD_SUCCESS")
print(f"Features: {feature_cols}")
```

## Ключевые правила

1. id_col — уникальный идентификатор строки (row_id), НЕ user_id.
   Мёрдж в train/test делай по user_id или product_id, а НЕ по id_col.

2. Агрегаты считай ТОЛЬКО на вспомогательных таблицах (users_df, orders_df и т.д.),
   маппируй в train и test через .map() или .merge(on='user_id'/'product_id').

3. НЕТ DATA LEAKAGE — любые target encoding считай только на train.

4. Обработка 1:N — сначала сгруппируй и сделай agg, потом мёрдж.
   После мёрджа len(train) не должен меняться.

## Обработка ошибок

Получишь код + traceback. Чини только сломанное место. Не переписывай всё.
"""


def _load_csv(path: str, sep: str = ",") -> pd.DataFrame:
    """Загружает CSV с автоопределением разделителя."""
    try:
        return pd.read_csv(path, sep=sep)
    except Exception:
        return pd.read_csv(path)


def _get_table_columns(data_dir: str, analyst_report: dict[str, Any]) -> dict[str, list[str]]:
    """Возвращает словарь {имя_таблицы: [колонки]} для всех таблиц из отчёта аналитика."""
    import os
    result: dict[str, list[str]] = {}
    tables = analyst_report.get("tables", {})
    # analyst может вернуть tables как список строк или как dict
    if isinstance(tables, list):
        tables = {t: {} for t in tables}
    for tname, tinfo in tables.items():
        sep = tinfo.get("separator", ",") if isinstance(tinfo, dict) else ","
        fpath = os.path.join(data_dir, tname)
        if os.path.exists(fpath):
            try:
                df = pd.read_csv(fpath, nrows=0, sep=sep)
                result[tname] = list(df.columns)
            except Exception:
                pass
    return result


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
    data_dir = "data/"
    output_dir = "output/"

    # Загружаем train/test заранее и передаём в exec — LLM не читает файлы сам
    logger.info("[coder:build_features] Загрузка train.csv и test.csv...")
    train = _load_csv(f"{data_dir}train.csv")
    test = _load_csv(f"{data_dir}test.csv")
    logger.info("[coder:build_features] train=%s, test=%s", train.shape, test.shape)

    # Собираем реальные колонки всех таблиц для подсказки в промпте
    table_columns = _get_table_columns(data_dir, analyst_report)
    # Добавляем train/test если их нет в отчёте
    table_columns.setdefault("train.csv", list(train.columns))
    table_columns.setdefault("test.csv", list(test.columns))

    cols_hint = "\n".join(
        f"  {tname}: {cols}" for tname, cols in table_columns.items()
    )

    features_json = json.dumps(feature_descriptions, ensure_ascii=False, indent=2)
    report_json = json.dumps(analyst_report, ensure_ascii=False, indent=2)

    # Загружаем вспомогательные таблицы и передаём их в exec_globals
    # чтобы LLM не ошибался с путями и разделителями
    aux_tables: dict[str, pd.DataFrame] = {}
    for tname in table_columns:
        if tname in ("train.csv", "test.csv"):
            continue
        fpath = f"{data_dir}{tname}"
        try:
            aux_tables[tname] = _load_csv(fpath)
        except Exception as e:
            logger.warning("[coder:build_features] Не удалось загрузить %s: %s", tname, e)

    # Подсказка: какие переменные уже загружены в exec_globals
    aux_vars_hint = "\n".join(
        f"  {tname.replace('.csv', '_df')} — pd.DataFrame, колонки: {list(df.columns)}"
        for tname, df in aux_tables.items()
    )

    user_message = (
        f"## Отчёт аналитика\n```json\n{report_json}\n```\n\n"
        f"## Признаки для реализации\n```json\n{features_json}\n```\n\n"
        f"## ВАЖНО: структура данных\n"
        f"  id_col='{id_col}' — уникальный идентификатор строки (НЕ пользователя!)\n"
        f"  train содержит колонки: {list(train.columns)}\n"
        f"  test  содержит колонки: {list(test.columns)}\n"
        f"  Признаки нужно считать через user_id/product_id во вспомогательных таблицах,\n"
        f"  затем мёрджить в train/test по user_id или product_id (НЕ по id_col='{id_col}').\n"
        f"  Финальный df_train должен содержать колонки: [{id_col}, {target_col}, feat_1..feat_N]\n"
        f"  Финальный df_test  должен содержать колонки: [{id_col}, feat_1..feat_N]\n\n"
        f"## Переменные уже доступны в коде (НЕ переопределяй, НЕ читай заново):\n"
        f"  id_col = '{id_col}'\n"
        f"  target_col = '{target_col}'\n"
        f"  data_dir = '{data_dir}'\n"
        f"  train — pd.DataFrame {train.shape}, колонки: {list(train.columns)}\n"
        f"  test  — pd.DataFrame {test.shape}, колонки: {list(test.columns)}\n"
        + (f"{aux_vars_hint}\n" if aux_vars_hint else "")
        + f"\n## Реальные колонки всех таблиц (используй точные имена):\n{cols_hint}\n\n"
        "Напиши код. Все таблицы уже загружены — НЕ читай файлы через pd.read_csv.\n"
        "Используй переменные напрямую: train, test, "
        + ", ".join(f"{t.replace('.csv', '_df')}" for t in aux_tables)
        + ".\nРезультат — переменные `df_train` и `df_test`."
    )

    messages = [
        {"role": "system", "content": BUILD_FEATURES_PROMPT},
        {"role": "user", "content": user_message},
    ]

    last_error = ""
    for attempt in range(1, MAX_RETRIES + 1):
        logger.info("[coder:build_features] Попытка %d/%d — запрос к GigaChat", attempt, MAX_RETRIES)

        if attempt > 1:
            aux_names = ", ".join(f"{t.replace('.csv', '_df')}" for t in aux_tables)
            messages.append({"role": "user", "content": (
                f"Код упал с ошибкой:\n{last_error}\n\n"
                "Исправь код. Верни только исправленный Python-код без объяснений.\n"
                f"Напомню: переменные уже существуют — train, test, id_col='{id_col}', "
                f"target_col='{target_col}', data_dir='{data_dir}'.\n"
                f"Вспомогательные таблицы: {aux_names}.\n"
                f"НЕ читай файлы через pd.read_csv — используй переменные напрямую.\n"
                f"train содержит колонки: {list(train.columns)}\n"
                f"test  содержит колонки: {list(test.columns)}\n"
                "Мёрдж в train/test делай по user_id или product_id, НЕ по id_col."
            )})

        response = with_retry(llm.invoke, messages)
        code = _extract_code(response.content)
        logger.info("[coder:build_features] Получен код (%d строк), исполняю...", code.count("\n"))
        messages.append({"role": "assistant", "content": response.content})

        exec_globals: dict[str, Any] = {
            "id_col": id_col,
            "target_col": target_col,
            "data_dir": data_dir,
            "output_dir": output_dir,
            "train": train.copy(),
            "test": test.copy(),
            "analysis_report": analyst_report,
            "separator": ",",
            "pd": pd,
            "np": np,
            "__builtins__": __builtins__,
            # Вспомогательные таблицы доступны как <имя_без_.csv>_df
            **{tname.replace(".csv", "_df"): df.copy() for tname, df in aux_tables.items()},
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
