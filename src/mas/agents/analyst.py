from __future__ import annotations

from pydantic import BaseModel, Field

from mas.agents.base import clip_text, gigachat_configured
from mas.domain.state import PipelineState
from mas.llm.client import build_gigachat
from mas.services import eda, feature_config, io, preprocessing
from mas.services.feature_catalog import build_catalog_ids
from mas.settings import Settings


class SchemaDecision(BaseModel):
    id_column: str = Field(description="Имя колонки идентификатора, общей для train и test")
    target_column: str | None = Field(
        default=None,
        description="Имя бинарного таргета в train; null если нет отдельной колонки",
    )
    rationale: str = ""


class AnalystAgent:
    """Агент 1: readme.txt + train/test, схема, EDA и заглушки (n,m,k)."""

    def __init__(self, settings: Settings) -> None:
        self._settings = settings

    def run(self, state: PipelineState, *, deadline: float) -> PipelineState:
        data_dir = state.repo_root / "data"
        readme_path, readme_text = io.read_readme(data_dir)
        train_in, test_in = io.read_train_test(data_dir)

        schema = self._schema_decision(readme_text, train_in, test_in, deadline)
        hid, htgt = io.infer_id_and_target(train_in, test_in)
        id_col = (
            schema.id_column
            if schema.id_column in train_in.columns and schema.id_column in test_in.columns
            else hid
        )
        tgt = schema.target_column
        if not tgt or tgt not in train_in.columns:
            tgt = htgt

        cleaned = preprocessing.basic_clean(train_in.copy())
        eda_map = eda.eda_all_columns(cleaned)
        configs = feature_config.compute_all_stub_configs(cleaned)
        catalog = build_catalog_ids(train_in, id_col, tgt)

        transcripts = dict(state.transcripts)
        transcripts["agent1"] = schema.rationale

        return state.model_copy(
            update={
                "readme_path": readme_path,
                "readme_text": readme_text,
                "id_column": id_col,
                "target_column": tgt,
                "train_in": train_in,
                "test_in": test_in,
                "eda": eda_map,
                "feature_configs": configs,
                "candidate_feature_ids": catalog,
                "transcripts": transcripts,
            }
        )

    def _schema_decision(self, readme: str, train, test, deadline: float) -> SchemaDecision:
        hid, htgt = io.infer_id_and_target(train, test)
        if not gigachat_configured():
            return SchemaDecision(id_column=hid, target_column=htgt, rationale="no credentials")

        llm = build_gigachat(
            self._settings, request_timeout=self._settings.remaining_llm_timeout(deadline)
        )
        prompt = (
            "Определи id_column (общая для train и test) и target_column (только в train, бинарная). "
            "Если таргет неочевиден — верни target_column=null.\n\n"
            f"readme.txt:\n{clip_text(readme)}\n\n"
            f"train columns: {list(train.columns)}\n"
            f"test columns: {list(test.columns)}"
        )
        try:
            chain = llm.with_structured_output(SchemaDecision)
            return chain.invoke(prompt)
        except Exception:
            return SchemaDecision(
                id_column=hid,
                target_column=htgt,
                rationale="llm_fallback",
            )
