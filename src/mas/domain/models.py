"""Модели для EDA/подготовки данных (первый агент)."""
from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass
class EDASummary:
    column: str
    dtype: str
    non_null_count: int
    null_ratio: float
    notes: str


@dataclass
class DatetimeColumnStats:
    by_month: pd.DataFrame
    by_weekday: pd.DataFrame
    trend_slope: float
    trend_r2: float
    trend_p: float
    has_trend: bool


@dataclass
class PrepareDataReport:
    duplicates_dropped: int
    constant_cols_dropped: list[str]
    high_cardinality_cols_dropped: list[str]
    class_balance: pd.DataFrame
    distributions: pd.DataFrame
    datetime_info: dict[str, DatetimeColumnStats]
    basic_eda: pd.DataFrame
