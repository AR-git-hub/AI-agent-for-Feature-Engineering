from __future__ import annotations

from pathlib import Path

import pandas as pd
from pydantic import BaseModel, ConfigDict, Field

from mas.domain.models import EDASummary, FeatureConfig


class PipelineState(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    repo_root: Path
    readme_path: Path | None = None
    readme_text: str = ""
    id_column: str = ""
    target_column: str | None = None
    train_in: pd.DataFrame | None = None
    test_in: pd.DataFrame | None = None
    train_enriched: pd.DataFrame | None = None
    test_enriched: pd.DataFrame | None = None
    eda: dict[str, EDASummary] = Field(default_factory=dict)
    feature_configs: dict[str, FeatureConfig] = Field(default_factory=dict)
    candidate_feature_ids: list[str] = Field(default_factory=list)
    new_feature_names: list[str] = Field(default_factory=list)
    top_feature_names: list[str] = Field(default_factory=list)
    transcripts: dict[str, str] = Field(default_factory=dict)
