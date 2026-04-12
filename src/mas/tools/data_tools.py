"""Тулы агента 1: чтение таблиц из data/ и request_merge с диагностикой."""
from __future__ import annotations

import json
import logging
from pathlib import Path

import pandas as pd

logger = logging.getLogger("mas.tools.data")

# ------------------------------------------------------------------
# Реестр таблиц — заполняется через init_registry(), используется
# request_merge и _load_table. Не хранит состояние между прогонами.
# ------------------------------------------------------------------
_TABLE_REGISTRY: dict[str, pd.DataFrame] = {}


def init_registry(tables: dict[str, pd.DataFrame]) -> None:
    """Инициализировать реестр из словаря загруженных таблиц (ctx.tables)."""
    _TABLE_REGISTRY.clear()
    _TABLE_REGISTRY.update(tables)


def register_merged_table(name: str, df: pd.DataFrame) -> None:
    """Сохранить результат мержа в реестр под именем `name`."""
    _TABLE_REGISTRY[name] = df


def _load_table(name: str) -> pd.DataFrame:
    if name not in _TABLE_REGISTRY:
        raise FileNotFoundError(f"Таблица '{name}' не найдена в реестре")
    return _TABLE_REGISTRY[name]


def _safe_json(data: dict) -> str:
    def _default(obj):
        if isinstance(obj, float) and (obj != obj):  # NaN
            return None
        return str(obj)
    return json.dumps(data, ensure_ascii=False, default=_default)


# ------------------------------------------------------------------
# Основные тулы чтения
# ------------------------------------------------------------------

def read_readme(data_dir: Path) -> str:
    path = data_dir / "readme.txt"
    if not path.exists():
        return ""
    return path.read_text(encoding="utf-8", errors="replace")


def list_csv_tables(data_dir: Path) -> list[str]:
    return sorted(p.name for p in data_dir.glob("*.csv"))


def load_csv(data_dir: Path, name: str, **read_csv_kwargs) -> pd.DataFrame:
    return pd.read_csv(data_dir / name, **read_csv_kwargs)


# ------------------------------------------------------------------
# request_merge
# ------------------------------------------------------------------

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
    После вызова результат можно исследовать через _load_table по result_name.
    """
    logger.info(
        "[request_merge] %s [%s] JOIN %s [%s], how='%s', result='%s'",
        left_table, left_key, right_table, right_key, how, result_name,
    )
    try:
        left = _load_table(left_table)
        right = _load_table(right_table)
    except FileNotFoundError as e:
        logger.error("[request_merge] Ошибка загрузки таблицы: %s", e)
        return json.dumps({"error": str(e)})

    left_rows_before = len(left)
    logger.info(
        "[request_merge] Строк до мержа: left=%d, right=%d", left_rows_before, len(right)
    )

    try:
        merged = pd.merge(
            left, right,
            left_on=left_key, right_on=right_key,
            how=how,
            suffixes=("", "_right"),
        )
    except Exception as e:
        logger.error("[request_merge] Ошибка pd.merge: %s", e)
        return json.dumps({"error": f"Ошибка при мерже: {e}"})

    # Удаляем дублирующий ключ из правой таблицы, если он отличается
    if right_key != left_key and right_key in merged.columns:
        merged = merged.drop(columns=[right_key])

    rows_after = len(merged)
    rows_lost = left_rows_before - rows_after if how == "left" else None
    match_rate = round(merged[left_key].notna().mean() * 100, 2) if left_key in merged.columns else None

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
        logger.warning(
            "[request_merge] Новые колонки с >50%% пропусков: %s",
            {k: v for k, v in null_in_new.items() if v > 50},
        )

    return _safe_json({
        "result_name": result_name,
        "left_rows_before": left_rows_before,
        "rows_after": rows_after,
        "rows_lost": rows_lost,
        "match_rate_pct": match_rate,
        "shape": {"rows": rows_after, "cols": merged.shape[1]},
        "new_columns": new_cols,
        "null_pct_in_new_columns": null_in_new,
        "duplicates_in_result": int(merged.duplicated().sum()),
    })
