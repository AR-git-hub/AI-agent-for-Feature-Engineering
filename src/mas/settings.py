from __future__ import annotations

import time

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

from mas.paths import repo_root


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=str(repo_root() / ".env"),
        env_file_encoding="utf-8",
        extra="ignore",
    )

    gigachat_model: str = Field(default="GigaChat-2-Max", validation_alias="GIGACHAT_MODEL")
    pipeline_time_budget_sec: float = Field(default=580.0, validation_alias="MAS_PIPELINE_BUDGET_SEC")
    llm_request_timeout_sec: float = Field(default=120.0, validation_alias="MAS_LLM_TIMEOUT_SEC")
    max_new_features: int = Field(default=5, ge=1, le=5, validation_alias="MAS_MAX_FEATURES")

    def monotonic_deadline(self) -> float:
        return time.monotonic() + float(self.pipeline_time_budget_sec)

    def remaining_llm_timeout(self, deadline: float) -> float:
        left = deadline - time.monotonic()
        cap = float(self.llm_request_timeout_sec)
        return max(15.0, min(cap, left - 5.0))
