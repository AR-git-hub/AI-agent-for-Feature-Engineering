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
Ты — агент-кодер. Пишешь Python-код для построения признаков. Ничего не придумываешь сам — реализуешь ТЗ генератора.

## Формат ответа

Верни ТОЛЬКО Python-код (без markdown, без пояснений). Используй только pandas и numpy.
Все переменные уже существуют. НЕ читай файлы через pd.read_csv.

---

## КРИТИЧЕСКИ ВАЖНО: работа со строковыми категориями

В данных МОГУТ быть строковые колонки (например month='may', day_of_week='mon', job='admin').
Прямое `.astype(int)` или арифметика над ними падает с ValueError.

### Правила обработки категориальных колонок

Если колонка типа object/string — ВСЕГДА делай одно из:

1. **Частотное кодирование (freq encoding)** — безопасно, работает всегда:
   ```python
   freq_map = client_data_df['job'].value_counts(normalize=True).to_dict()
   train_work['job_freq'] = train_work['client_id'].map(
       client_data_df.set_index('client_id')['job']
   ).map(freq_map).fillna(0)
   ```

2. **Target encoding** — только по train, потом маппировать на test:
   ```python
   # Соединяем train с client_data чтобы получить категорию
   train_with_cat = train.merge(client_data_df[['client_id','job']], on='client_id', how='left')
   te_map = train_with_cat.groupby('job')[target_col].mean().to_dict()
   global_mean = train[target_col].mean()
   train_work['job_te'] = train_with_cat['job'].map(te_map).fillna(global_mean)
   test_with_cat = test.merge(client_data_df[['client_id','job']], on='client_id', how='left')
   test_work['job_te'] = test_with_cat['job'].map(te_map).fillna(global_mean)
   ```

3. **Явный маппинг упорядоченных категорий** через dict:
   ```python
   MONTH_MAP = {'jan':1,'feb':2,'mar':3,'apr':4,'may':5,'jun':6,
                'jul':7,'aug':8,'sep':9,'oct':10,'nov':11,'dec':12}
   DOW_MAP = {'mon':0,'tue':1,'wed':2,'thu':3,'fri':4,'sat':5,'sun':6}
   # использовать через .map(MONTH_MAP) — результат числовой, можно брать sin/cos
   ```

### НИКОГДА не делай

- `pd.to_numeric(series)` на строковой колонке без errors='coerce'
- `.astype(int)` / `.astype(float)` на строках типа 'may','mon'
- Арифметику и sin/cos поверх необработанных строк

---

## ОБЯЗАТЕЛЬНЫЙ ШАБЛОН КОДА

```python
import numpy as np
import pandas as pd

original_train_len = len(train)
original_test_len = len(test)

train_work = train.copy()
test_work  = test.copy()

# Шаг 1: Собрать признаки в временные Series/DataFrame
# Используй .map() / .merge() для привязки к train/test по ключу (например client_id).
# Категориальные колонки обрабатывай строго через freq/target encoding или явный dict.

# Пример 1 — числовой признак из client_data:
# num_map = client_data_df.set_index('client_id')['age']
# train_work['age_feat'] = train_work['client_id'].map(num_map)
# test_work['age_feat']  = test_work['client_id'].map(num_map)

# Пример 2 — freq encoding категории job:
# job_freq = client_data_df['job'].value_counts(normalize=True).to_dict()
# train_work['job_freq'] = train_work['client_id'].map(
#     client_data_df.set_index('client_id')['job']).map(job_freq).fillna(0)
# test_work['job_freq']  = test_work['client_id'].map(
#     client_data_df.set_index('client_id')['job']).map(job_freq).fillna(0)

# Шаг 2: перечисли имена признаков
feature_cols = ['feat_1', 'feat_2']  # ЗАМЕНИ на реальные имена

# Шаг 3: Заполни NaN/inf медианой по train (для числовых) или 0 (для всего остального)
for col in feature_cols:
    s_train = pd.to_numeric(train_work[col], errors='coerce')
    s_test  = pd.to_numeric(test_work[col],  errors='coerce')
    s_train = s_train.replace([np.inf, -np.inf], np.nan)
    s_test  = s_test.replace([np.inf, -np.inf], np.nan)
    med = s_train.median()
    fill = float(med) if not pd.isna(med) else 0.0
    train_work[col] = s_train.fillna(fill).astype(float)
    test_work[col]  = s_test.fillna(fill).astype(float)

# Шаг 4: Собрать итоговые df_train / df_test
df_train = train_work[[id_col, target_col] + feature_cols].copy()
df_test  = test_work[[id_col] + feature_cols].copy()

assert len(df_train) == original_train_len
assert len(df_test)  == original_test_len
assert df_train[feature_cols].isna().sum().sum() == 0
assert df_test[feature_cols].isna().sum().sum() == 0
print("BUILD_SUCCESS")
print(f"Features: {feature_cols}")
```

---

## Ключевые правила

1. **id_col** — идентификатор строки для сохранения. Мёрдж с доп.таблицами делай по тому же полю,
   которое выступает ключом (часто это и есть client_id = id_col, или user_id — смотри отчёт аналитика).

2. **Размер не меняется**: `len(df_train) == original_train_len`, `len(df_test) == original_test_len`.
   Если связь 1:N — сначала groupby+agg на доп.таблице, потом map/merge.

3. **Нет NaN в финале**: заполняй всё медианой/нулём. Проверяй assert.

4. **Нет data leakage**: target encoding считай ТОЛЬКО по train. На test — маппируй готовую таблицу.

5. **Все итоговые признаки — числовые (float)**: CatBoost принимает строки как cat_features,
   но сохранённые признаки лучше кодировать числом, чтобы compute_stats/scoring работали стабильно.

6. **Максимум 5 признаков**: ровно столько, сколько предложил генератор (≤5).

## Обработка ошибок

Получишь код + traceback. Чини только сломанное место. Если ошибка была на строковом значении
(например `can't multiply sequence by non-int`, `could not convert string to float`) — значит
ты пытался делать арифметику на строковой колонке. Переделай через freq/target encoding.
"""


def _load_csv(path: str, sep: str = ",") -> pd.DataFrame:
    """Загружает CSV с автоопределением разделителя."""
    try:
        return pd.read_csv(path, sep=sep)
    except Exception:
        return pd.read_csv(path)


def _get_table_columns(data_dir: str, analyst_report: dict[str, Any]) -> dict[str, list[str]]:
    """Возвращает словарь {имя_таблицы: [колонки]} для всех таблиц из отчёта аналитика.

    Поддерживает разные форматы tables в отчёте:
      - list[dict] (новый формат): [{"name": "x.csv", "separator": ",", ...}]
      - list[str]: ["x.csv", "y.csv"]
      - dict[str, dict]: {"x.csv": {"separator": ","}}
    """
    import os
    result: dict[str, list[str]] = {}
    tables_raw = analyst_report.get("tables", {})

    # Нормализуем в dict[str, dict]
    tables: dict[str, dict] = {}
    if isinstance(tables_raw, list):
        for item in tables_raw:
            if isinstance(item, dict):
                name = item.get("name") or item.get("table") or item.get("file")
                if name:
                    tables[name] = item
            elif isinstance(item, str):
                tables[item] = {}
    elif isinstance(tables_raw, dict):
        for k, v in tables_raw.items():
            tables[k] = v if isinstance(v, dict) else {}

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

    # Страховка: подхватываем все CSV из data/ даже если аналитик про них забыл/
    # передал tables в неожиданном формате. Критично, чтобы aux_tables были доступны в exec.
    import os as _os
    try:
        for fname in _os.listdir(data_dir):
            if fname.lower().endswith(".csv") and fname not in table_columns:
                try:
                    df_head = pd.read_csv(f"{data_dir}{fname}", nrows=0)
                    table_columns[fname] = list(df_head.columns)
                    logger.info("[coder:build_features] Авто-подхват таблицы: %s", fname)
                except Exception as e:
                    logger.warning("[coder:build_features] Не удалось прочитать %s: %s", fname, e)
    except Exception as e:
        logger.warning("[coder:build_features] Не удалось сканировать %s: %s", data_dir, e)

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
        raw_content = response.content if isinstance(response.content, str) else str(response.content)
        code = _extract_code(raw_content)
        logger.info("[coder:build_features] Получен код (%d строк), исполняю...", code.count("\n"))
        messages.append({"role": "assistant", "content": raw_content})

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

    y = np.asarray(df_train[target_col].values)
    stats: dict[str, dict[str, Any]] = {}

    # Подготавливаем числовую матрицу
    X_encoded = pd.DataFrame(index=df_train.index)
    for col in feature_cols:
        series = df_train[col]
        if pd.api.types.is_numeric_dtype(series):
            X_encoded[col] = pd.to_numeric(series, errors="coerce").fillna(-999).astype(float)
        else:
            le = LabelEncoder()
            X_encoded[col] = le.fit_transform(series.astype(str).fillna("__null__"))

    # Mutual information
    try:
        mi_scores = mutual_info_classif(X_encoded.values, y, random_state=42)
    except Exception as e:
        logger.warning("[coder:compute_stats] mutual_info_classif ошибка: %s", e)
        mi_scores = np.zeros(len(feature_cols))

    for i, col in enumerate(feature_cols):
        series = df_train[col]
        col_enc = np.asarray(X_encoded[col].values, dtype=float)

        null_pct = round(float(series.isna().mean()) * 100, 2)
        nunique = int(series.nunique())

        # Pearson
        try:
            if np.std(col_enc) == 0 or np.std(y) == 0:
                pearson = 0.0
            else:
                pearson = round(float(np.corrcoef(col_enc, y.astype(float))[0, 1]), 4)
                if np.isnan(pearson):
                    pearson = 0.0
        except Exception:
            pearson = 0.0

        # Spearman — scipy >=1.9 возвращает SignificanceResult с .statistic,
        # старые версии — именованный кортеж с .correlation. Извлекаем безопасно.
        try:
            sp_res = spearmanr(col_enc, y)
            val = getattr(sp_res, "statistic", None)
            if val is None:
                val = getattr(sp_res, "correlation", None)
            if val is None:
                try:
                    val = sp_res[0]  # type: ignore[index]
                except Exception:
                    val = 0.0
            spearman_f = float(np.asarray(val, dtype=float))
            spearman = 0.0 if np.isnan(spearman_f) else round(spearman_f, 4)
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
            # loc[row_label, list_of_cols] — валидный pandas, но у некоторых stubs
            # возникают ошибки типов; приводим явно к float через numpy.
            row_vals = np.asarray(corr_matrix.loc[col][other_cols].values, dtype=float)
            row_vals = row_vals[~np.isnan(row_vals)]
            max_corr = float(row_vals.max()) if row_vals.size else 0.0
            stats[col]["max_corr_with_others"] = round(max_corr, 4)
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
