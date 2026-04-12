"""Общий контекст прогона, который передаётся между агентами."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


@dataclass
class RunContext:
    """Состояние пайплайна: датасет → фичи и метрики → отбор → артефакты."""

    data_dir: Path
    """Каталог с `train.csv` / `test.csv` / `readme.txt` — источник для обучения/прогона (обычно `data/`)."""
    output_dir: Path
    """
    Куда агент 4 записывает результаты.
    Ожидаемый формат файлов (вход для CatBoost):
      output/train.csv — id объекта, целевая переменная, ≤5 сгенерированных признаков.
      output/test.csv  — id объекта, те же ≤5 признаков (без таргета).
    """
    configs_dir: Path | None = None
    """Каталог с тестовыми фикстурами (`configs/`). Агенты не читают его — только pytest/тесты."""

    # --- Агент 1: датасет ---
    readme_text: str = ""
    input_train_cols: list[str] = field(default_factory=list)
    """Исходные колонки data/train.csv — обязательны в output/train.csv."""
    input_test_cols: list[str] = field(default_factory=list)
    """Исходные колонки data/test.csv — обязательны в output/test.csv."""
    tables: dict[str, pd.DataFrame] = field(default_factory=dict)
    schema_notes: str = ""
    """Черновые заметки о ключах/джойнах (LLM или эвристики)."""
    eda_basic_by_table: dict[str, pd.DataFrame] = field(default_factory=dict)
    """Результат `eda_tools.basic_eda` по каждой загруженной таблице (имя файла → отчёт)."""
    merge_reports: list[str] = field(default_factory=list)
    """JSON-диагностика каждого вызова `request_merge` (match_rate, потери строк, null% в новых колонках)."""
    eda_report: str = ""
    """Компактный текстовый EDA-отчёт по признакам train_frame — передаётся в промпт агента 2."""
    target_col: str | None = None
    """Имя целевой переменной — определяется как колонка, присутствующая в train, но не в test."""
    id_col: str | None = None
    """Имя id-колонки объекта — идёт в output/, но НЕ в матрицу фичей."""
    train_frame: pd.DataFrame | None = None
    """Итоговая train-таблица после сборки и обработки (объекты + таргет + базовые колонки)."""
    test_frame: pd.DataFrame | None = None
    """Итоговая test-таблица в том же пространстве признаков, что и train (без таргета)."""

    # --- Агент 2: генерация фичей и метрики ---
    features_eda_report: str = ""
    """EDA-отчёт по сгенерированным фичам (с corr_target и importance) — передаётся в промпт агента 3."""
    feature_matrix_train: pd.DataFrame | None = None
    """Сгенерированные признаки для train (только новые колонки или полный блок — на ваш выбор)."""
    feature_matrix_test: pd.DataFrame | None = None
    feature_column_names: list[str] = field(default_factory=list)
    """Порядок имён фичей; совпадает с осями metric_m_matrix."""
    metric_m_matrix: np.ndarray | None = None
    """Матрица n×n: metric_m(feature_i, feature_j) в ячейке (i, j)."""
    selection_prompt: str = ""
    """Промпт для агента отбора (агент 3), уже с подставленными метриками."""

    # --- Агент 3: отбор ---
    selection_notes: str = ""
    """Пояснения агента отбора (цепочка мыслей, правила)."""
    selected_feature_names: list[str] = field(default_factory=list)

    # --- Агент 4: ответ ---
    train_features: pd.DataFrame | None = None
    test_features: pd.DataFrame | None = None
    """Финальные таблицы для записи в output/ (id, target?, ≤5 фич)."""

    # совместимость / отладка
    feature_candidates: list[dict[str, Any]] = field(default_factory=list)
