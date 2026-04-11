from __future__ import annotations

from pydantic import BaseModel, Field

from mas.agents.base import clip_text, gigachat_configured
from mas.domain.state import PipelineState
from mas.llm.client import build_gigachat
from mas.settings import Settings


class TopFeatureSelection(BaseModel):
    names: list[str] = Field(default_factory=list, description="1..5 имён новых колонок")
    rationale: str = ""


class SelectorAgent:
    """Агент 3: выбор 1..5 финальных признаков только среди сгенерированных."""

    def __init__(self, settings: Settings) -> None:
        self._settings = settings

    def run(self, state: PipelineState, *, deadline: float) -> PipelineState:
        if state.train_enriched is None:
            raise RuntimeError("Agent3: нет train_enriched")

        allowed = [c for c in state.new_feature_names if c in state.train_enriched.columns]
        if not allowed:
            raise RuntimeError("Agent3: пустой список новых признаков")

        sel = self._select(state, allowed, deadline)
        picked = [n for n in sel.names if n in allowed][: self._settings.max_new_features]
        if len(picked) < 1:
            picked = allowed[: self._settings.max_new_features]

        transcripts = dict(state.transcripts)
        transcripts["agent3"] = sel.rationale

        return state.model_copy(
            update={
                "top_feature_names": picked,
                "transcripts": transcripts,
            }
        )

    def _select(
        self, state: PipelineState, allowed: list[str], deadline: float
    ) -> TopFeatureSelection:
        mx = self._settings.max_new_features
        if not gigachat_configured():
            return TopFeatureSelection(names=allowed[:mx], rationale="no credentials")

        llm = build_gigachat(
            self._settings, request_timeout=self._settings.remaining_llm_timeout(deadline)
        )
        eda_blob = "\n".join(
            f"{k}: {v.model_dump()}" for k, v in state.eda.items() if k in allowed
        )
        cfg_blob = "\n".join(
            f"{k}: {state.feature_configs[k].model_dump()}"
            for k in allowed
            if k in state.feature_configs
        )
        prompt = (
            f"Выбери от 1 до {mx} признаков для бинарной классификации. Только из списка allowed, "
            "сохрани порядок важности (лучшие первыми).\n\n"
            f"README:\n{clip_text(state.readme_text)}\n\n"
            f"EDA по кандидатам:\n{clip_text(eda_blob, 12000)}\n\n"
            f"Конфиги (n,m,k заглушки):\n{clip_text(cfg_blob, 6000)}\n\n"
            f"allowed: {allowed}"
        )
        try:
            chain = llm.with_structured_output(TopFeatureSelection)
            out = chain.invoke(prompt)
            cleaned = [x for x in out.names if x in allowed][:mx]
            if not cleaned:
                cleaned = allowed[:mx]
            return TopFeatureSelection(names=cleaned, rationale=out.rationale)
        except Exception:
            return TopFeatureSelection(names=allowed[:mx], rationale="llm_fallback")
