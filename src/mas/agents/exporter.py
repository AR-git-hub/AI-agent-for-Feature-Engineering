from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from mas.domain.state import PipelineState
from mas.settings import Settings


def _sanitize_feature_column(s: pd.Series) -> pd.Series:
    if s.isna().all():
        return pd.Series(np.zeros(len(s)), index=s.index, dtype=float)
    s = s.fillna(s.median())
    if s.isna().all():
        return pd.Series(np.zeros(len(s)), index=s.index, dtype=float)
    return s


class ExporterAgent:
    """Агент 4: записывает output/train.csv и output/test.csv (все исходные + выбранные признаки)."""

    def __init__(self, settings: Settings) -> None:
        self._settings = settings

    def run(self, state: PipelineState, *, deadline: float) -> PipelineState:
        _ = deadline
        if state.train_in is None or state.test_in is None:
            raise RuntimeError("Agent4: нет исходных таблиц")
        if state.train_enriched is None or state.test_enriched is None:
            raise RuntimeError("Agent4: нет обогащённых таблиц")
        if not state.top_feature_names:
            raise RuntimeError("Agent4: нет top_feature_names")

        out_dir = state.repo_root / "output"
        out_dir.mkdir(parents=True, exist_ok=True)

        out_train = state.train_in.copy()
        out_test = state.test_in.copy()
        for col in state.top_feature_names:
            out_train[col] = _sanitize_feature_column(state.train_enriched[col].copy())
            out_test[col] = _sanitize_feature_column(state.test_enriched[col].copy())

        train_path = out_dir / "train.csv"
        test_path = out_dir / "test.csv"
        out_train.to_csv(train_path, index=False)
        out_test.to_csv(test_path, index=False)

        transcripts = dict(state.transcripts)
        transcripts["agent4"] = "written output/train.csv output/test.csv"

        return state.model_copy(update={"transcripts": transcripts})
