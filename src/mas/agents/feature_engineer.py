from __future__ import annotations

from pydantic import BaseModel, Field

import pandas as pd

from mas.agents.base import clip_text, gigachat_configured
from mas.domain.state import PipelineState
from mas.llm.client import build_gigachat
from mas.services import eda, feature_config, preprocessing
from mas.services.feature_catalog import apply_catalog_features
from mas.settings import Settings


class CatalogPick(BaseModel):
    feature_ids: list[str] = Field(default_factory=list, description="1..5 id из каталога")
    rationale: str = ""


class FeatureEngineerAgent:
    """Агент 2: выбор признаков из каталога и расчёт на train/test + пересчёт EDA/конфигов."""

    def __init__(self, settings: Settings) -> None:
        self._settings = settings

    def run(self, state: PipelineState, *, deadline: float) -> PipelineState:
        if state.train_in is None or state.test_in is None:
            raise RuntimeError("Agent2: нет train/test")

        pick = self._pick(state, deadline)
        mx = self._settings.max_new_features
        ids = [x for x in pick.feature_ids if x in state.candidate_feature_ids][:mx]
        while len(ids) < 1:
            ids = state.candidate_feature_ids[:1]

        tr_e, te_e, _ = apply_catalog_features(
            state.train_in,
            state.test_in,
            ids,
            state.id_column,
            state.target_column,
        )
        new_names = [c for c in tr_e.columns if c not in state.train_in.columns]

        eda_map = eda.eda_all_columns(preprocessing.basic_clean(tr_e.copy()))
        configs = feature_config.compute_all_stub_configs(tr_e)

        transcripts = dict(state.transcripts)
        transcripts["agent2"] = pick.rationale

        return state.model_copy(
            update={
                "train_enriched": tr_e,
                "test_enriched": te_e,
                "new_feature_names": new_names,
                "eda": eda_map,
                "feature_configs": configs,
                "transcripts": transcripts,
            }
        )

    def _pick(self, state: PipelineState, deadline: float) -> CatalogPick:
        catalog = state.candidate_feature_ids
        mx = self._settings.max_new_features
        if not catalog:
            return CatalogPick(feature_ids=[], rationale="empty_catalog")

        if not gigachat_configured():
            return CatalogPick(feature_ids=catalog[:mx], rationale="no credentials")

        llm = build_gigachat(
            self._settings, request_timeout=self._settings.remaining_llm_timeout(deadline)
        )
        eda_blob = "\n".join(f"{k}: {v.model_dump()}" for k, v in list(state.eda.items())[:80])
        prompt = (
            f"Выбери от 1 до {mx} элементов из каталога (ровно id строки). "
            "Только из списка catalog, без выдуманных id.\n\n"
            f"README:\n{clip_text(state.readme_text)}\n\n"
            f"EDA (фрагмент):\n{clip_text(eda_blob, 12000)}\n\n"
            f"catalog:\n{catalog}"
        )
        try:
            chain = llm.with_structured_output(CatalogPick)
            out = chain.invoke(prompt)
            cleaned = [x for x in out.feature_ids if x in catalog][:mx]
            if not cleaned:
                cleaned = catalog[:mx]
            return CatalogPick(feature_ids=cleaned, rationale=out.rationale)
        except Exception:
            return CatalogPick(feature_ids=catalog[:mx], rationale="llm_fallback")
