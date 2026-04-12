"""Тулы для LLM (function calling): по доменам в отдельных модулях."""
from __future__ import annotations

import importlib
from typing import TYPE_CHECKING

from src.mas.tools import data_tools, eda_tools, output_tools

if TYPE_CHECKING:
    from src.mas.tools import feature_tools as feature_tools

__all__ = ["data_tools", "eda_tools", "feature_tools", "output_tools"]


def __getattr__(name: str):
    # feature_tools тянет scoring — подгружаем только по запросу (run без CatBoost-скоринга не падает).
    if name == "feature_tools":
        return importlib.import_module("src.mas.tools.feature_tools")
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
