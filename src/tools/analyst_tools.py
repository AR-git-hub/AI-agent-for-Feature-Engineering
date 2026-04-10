"""
6 инструментов агента-аналитика.

Инструменты работают с файловой системой data/ и возвращают
структурированные строки (JSON), которые LLM использует как наблюдения.
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any

import pandas as pd
import numpy as np
from langchain_core.tools import tool

logger = logging.getLogger(__name__)

DATA_DIR = Path("data")

# Хранилище мерджей, которые кодер уже выполнил (имя → DataFrame).
# Аналитик может обращаться к ним через table_info / peek_rows по имени.
_merged_tables: dict[str, pd.DataFrame] = {}


def register_merged_table(name: str, df: pd.DataFrame) -> None:
    """Регистрирует результат мержа, чтобы аналитик мог его исследовать."""
    _merged_tables[name] = df


def _load_table(name: str) -> pd.DataFrame:
    """Загружает таблицу из data/ или из кэша мерджей."""
    if name in _merged_tables:
        return _merged_tables[name]
    path = DATA_DIR / name
    if not path.exists():
        # Попробуем добавить .csv
        path = DATA_DIR / (name + ".csv")
    if not path.exists():
        raise FileNotFoundError(f"Таблица '{name}' не найдена в {DATA_DIR}")
    # Пробуем разные разделители
    for sep in [",", ";", "\t", "|"]:
        try:
            df = pd.read_csv(path, sep=sep)
            if df.shape[1] > 1:
                return df
        except Exception:
            continue
    return pd.read_csv(path)


# ---------------------------------------------------------------------------
# Инструмент 1: read_readme
# ---------------------------------------------------------------------------

@tool
def read_readme() -> str:
    """Читает файл data/readme.txt и возвращает его содержимое.

    Это первый инструмент, который должен вызвать аналитик.
    Из readme он узнаёт: тип задачи, список таблиц, их схему,
    имя таргета и идентификатора.
    """
    logger.info("[read_readme] Чтение data/readme.txt")
    path = DATA_DIR / "readme.txt"
    if not path.exists():
        logger.warning("[read_readme] readme.txt не найден в data/")
        return json.dumps({"error": "readme.txt не найден в data/"})
    text = path.read_text(encoding="utf-8")
    logger.info("[read_readme] Прочитано %d символов", len(text))
    return json.dumps({"readme": text}, ensure_ascii=False)


# ---------------------------------------------------------------------------
# Инструмент 2: list_tables
# ---------------------------------------------------------------------------

@tool
def list_tables() -> str:
    """Сканирует папку data/ и возвращает список всех CSV-файлов с метаданными.

    Для каждого файла возвращает:
    - имя файла
    - количество строк и столбцов
    - размер файла в КБ
    - определённый разделитель
    - названия колонок
    """
    logger.info("[list_tables] Сканирование папки %s", DATA_DIR)
    result = []
    for fpath in sorted(DATA_DIR.glob("*.csv")):
        info: dict[str, Any] = {"file": fpath.name}
        info["size_kb"] = round(fpath.stat().st_size / 1024, 1)
        try:
            df = _load_table(fpath.name)
            info["rows"] = int(df.shape[0])
            info["cols"] = int(df.shape[1])
            info["columns"] = list(df.columns)
            logger.info("[list_tables]   %s: %d строк, %d колонок, %.1f КБ",
                        fpath.name, info["rows"], info["cols"], info["size_kb"])
        except Exception as e:
            logger.error("[list_tables]   %s: ошибка загрузки — %s", fpath.name, e)
            info["error"] = str(e)
        result.append(info)

    # Добавим зарегистрированные мерджи
    for name in _merged_tables:
        df = _merged_tables[name]
        result.append({
            "file": name,
            "source": "merged",
            "rows": int(df.shape[0]),
            "cols": int(df.shape[1]),
            "columns": list(df.columns),
        })
        logger.info("[list_tables]   %s (merged): %d строк, %d колонок", name, df.shape[0], df.shape[1])

    logger.info("[list_tables] Найдено таблиц: %d", len(result))
    return json.dumps({"tables": result}, ensure_ascii=False)


# ---------------------------------------------------------------------------
# Инструмент 3: table_info
# ---------------------------------------------------------------------------

@tool
def table_info(table_name: str) -> str:
    """Возвращает сжатую статистику по таблице: типы колонок, пропуски, уникальные значения, describe().

    Args:
        table_name: имя CSV-файла (например, "client_data.csv") или имя мержа.

    Не возвращает сами данные — только метаинформацию.
    """
    logger.info("[table_info] Загрузка статистики для '%s'", table_name)
    try:
        df = _load_table(table_name)
    except FileNotFoundError as e:
        logger.error("[table_info] Таблица '%s' не найдена: %s", table_name, e)
        return json.dumps({"error": str(e)})

    logger.info("[table_info] '%s': shape=%s", table_name, df.shape)
    info: dict[str, Any] = {
        "table": table_name,
        "shape": {"rows": int(df.shape[0]), "cols": int(df.shape[1])},
        "columns": {},
    }

    for col in df.columns:
        col_info: dict[str, Any] = {
            "dtype": str(df[col].dtype),
            "null_count": int(df[col].isna().sum()),
            "null_pct": round(float(df[col].isna().mean()) * 100, 2),
            "nunique": int(df[col].nunique()),
        }
        if pd.api.types.is_numeric_dtype(df[col]):
            desc = df[col].describe()
            col_info["min"] = round(float(desc["min"]), 4) if not pd.isna(desc["min"]) else None
            col_info["max"] = round(float(desc["max"]), 4) if not pd.isna(desc["max"]) else None
            col_info["mean"] = round(float(desc["mean"]), 4) if not pd.isna(desc["mean"]) else None
            col_info["std"] = round(float(desc["std"]), 4) if not pd.isna(desc["std"]) else None
        else:
            # Для категориальных — топ-5 значений
            top = df[col].value_counts().head(5)
            col_info["top_values"] = {str(k): int(v) for k, v in top.items()}
        info["columns"][col] = col_info

    # Дубликаты по всем колонкам
    info["duplicate_rows"] = int(df.duplicated().sum())

    high_null_cols = [c for c, v in info["columns"].items() if v["null_pct"] > 30]
    if high_null_cols:
        logger.warning("[table_info] '%s': колонки с пропусками >30%%: %s", table_name, high_null_cols)
    logger.info("[table_info] '%s': дубликатов строк=%d", table_name, info["duplicate_rows"])

    return json.dumps(info, ensure_ascii=False)


# ---------------------------------------------------------------------------
# Инструмент 4: peek_rows
# ---------------------------------------------------------------------------

@tool
def peek_rows(table_name: str, n: int = 5, mode: str = "head") -> str:
    """Показывает N строк из таблицы.

    Args:
        table_name: имя CSV-файла или мержа.
        n: количество строк (по умолчанию 5).
        mode: "head" — первые строки, "tail" — последние, "sample" — случайные.

    Используется для визуального понимания формата данных и обнаружения паттернов.
    """
    logger.info("[peek_rows] '%s': mode=%s, n=%d", table_name, mode, n)
    try:
        df = _load_table(table_name)
    except FileNotFoundError as e:
        logger.error("[peek_rows] Таблица '%s' не найдена: %s", table_name, e)
        return json.dumps({"error": str(e)})

    n = min(n, 50)  # защита от слишком больших выборок
    if mode == "head":
        rows = df.head(n)
    elif mode == "tail":
        rows = df.tail(n)
    elif mode == "sample":
        rows = df.sample(min(n, len(df)), random_state=42)
    else:
        logger.error("[peek_rows] Неизвестный mode='%s'", mode)
        return json.dumps({"error": f"Неизвестный mode='{mode}'. Используйте: head, tail, sample"})

    logger.info("[peek_rows] Возвращено %d строк из '%s'", len(rows), table_name)
    # Конвертируем NaN в None для JSON
    records = rows.where(pd.notnull(rows), None).to_dict(orient="records")
    return json.dumps(
        {"table": table_name, "mode": mode, "n": len(records), "rows": records},
        ensure_ascii=False,
        default=str,
    )


# ---------------------------------------------------------------------------
# Инструмент 5: request_merge
# ---------------------------------------------------------------------------

@tool
def request_merge(
    left_table: str,
    right_table: str,
    left_key: str,
    right_key: str,
    how: str = "left",
    result_name: str = "merged",
) -> str:
    """Выполняет мерж двух таблиц и регистрирует результат для дальнейшего исследования.

    Args:
        left_table: имя левой таблицы (например, "train.csv").
        right_table: имя правой таблицы (например, "client_data.csv").
        left_key: колонка-ключ в левой таблице.
        right_key: колонка-ключ в правой таблице.
        how: тип джойна — "left", "inner", "outer", "right" (по умолчанию "left").
        result_name: имя, под которым сохранить результат для последующего обращения.

    Возвращает диагностику: shape, потери строк, процент null в новых колонках.
    После вызова результат можно исследовать через table_info и peek_rows по result_name.
    """
    logger.info(
        "[request_merge] %s [%s] LEFT JOIN %s [%s], how='%s', result='%s'",
        left_table, left_key, right_table, right_key, how, result_name,
    )
    try:
        left = _load_table(left_table)
        right = _load_table(right_table)
    except FileNotFoundError as e:
        logger.error("[request_merge] Ошибка загрузки таблицы: %s", e)
        return json.dumps({"error": str(e)})

    left_rows_before = len(left)
    logger.info("[request_merge] Строк до мержа: left=%d, right=%d", left_rows_before, len(right))

    try:
        merged = pd.merge(left, right, left_on=left_key, right_on=right_key, how=how, suffixes=("", "_right"))
    except Exception as e:
        logger.error("[request_merge] Ошибка pd.merge: %s", e)
        return json.dumps({"error": f"Ошибка при мерже: {e}"})

    # Удаляем дублирующий ключ из правой таблицы, если он отличается
    if right_key != left_key and right_key in merged.columns:
        merged = merged.drop(columns=[right_key])

    rows_after = len(merged)
    rows_lost = left_rows_before - rows_after if how == "left" else None

    # Новые колонки (из правой таблицы)
    new_cols = [c for c in merged.columns if c not in left.columns]
    null_in_new = {
        col: round(float(merged[col].isna().mean()) * 100, 2)
        for col in new_cols
    }

    register_merged_table(result_name, merged)

    logger.info(
        "[request_merge] Результат '%s': shape=%s, потеряно строк=%s, новых колонок=%d",
        result_name, merged.shape, rows_lost, len(new_cols),
    )
    if any(v > 50 for v in null_in_new.values()):
        logger.warning("[request_merge] Новые колонки с >50%% пропусков: %s",
                       {k: v for k, v in null_in_new.items() if v > 50})

    return json.dumps({
        "result_name": result_name,
        "left_rows_before": left_rows_before,
        "rows_after": rows_after,
        "rows_lost": rows_lost,
        "shape": {"rows": rows_after, "cols": merged.shape[1]},
        "new_columns": new_cols,
        "null_pct_in_new_columns": null_in_new,
        "duplicates_in_result": int(merged.duplicated().sum()),
    }, ensure_ascii=False)


# ---------------------------------------------------------------------------
# Инструмент 6: report
# ---------------------------------------------------------------------------

@tool
def report(
    tables: list[str],
    id_column: str,
    target_column: str,
    key_relations: list[dict],
    numeric_columns: list[str],
    categorical_columns: list[str],
    datetime_columns: list[str],
    missing_issues: list[str],
    join_recommendations: list[str],
    potential_problems: list[str],
    notes: str = "",
) -> str:
    """Финальный инструмент аналитика. Формирует структурированный JSON-отчёт и завершает исследование.

    Args:
        tables: список использованных таблиц.
        id_column: имя колонки-идентификатора объекта.
        target_column: имя колонки-таргета.
        key_relations: список связей между таблицами, например:
            [{"left": "train.csv", "right": "client_data.csv", "left_key": "client_id", "right_key": "client_id"}]
        numeric_columns: числовые колонки в данных.
        categorical_columns: категориальные колонки в данных.
        datetime_columns: колонки с датами/временем.
        missing_issues: описание проблем с пропусками.
        join_recommendations: рекомендации по объединению таблиц для генератора.
        potential_problems: потенциальные проблемы (дубли, many-to-many, leakage и т.д.).
        notes: дополнительные наблюдения.

    Этот отчёт передаётся оркестратору и используется генератором и кодером.
    """
    logger.info(
        "[report] Формирование финального отчёта: таблиц=%d, числовых колонок=%d, "
        "категориальных=%d, дат=%d",
        len(tables), len(numeric_columns), len(categorical_columns), len(datetime_columns),
    )
    if potential_problems:
        logger.warning("[report] Потенциальные проблемы: %s", potential_problems)
    logger.info("[report] Отчёт сформирован. Исследование завершено.")

    analyst_report = {
        "tables": tables,
        "id_column": id_column,
        "target_column": target_column,
        "key_relations": key_relations,
        "columns": {
            "numeric": numeric_columns,
            "categorical": categorical_columns,
            "datetime": datetime_columns,
        },
        "missing_issues": missing_issues,
        "join_recommendations": join_recommendations,
        "potential_problems": potential_problems,
        "notes": notes,
    }
    return json.dumps({"analyst_report": analyst_report}, ensure_ascii=False)


# ---------------------------------------------------------------------------
# Экспорт списка инструментов
# ---------------------------------------------------------------------------

ANALYST_TOOLS = [
    read_readme,
    list_tables,
    table_info,
    peek_rows,
    request_merge,
    report,
]
