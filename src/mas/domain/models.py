from typing import Any

from pydantic import BaseModel, Field


class FeatureConfig(BaseModel):
    column: str
    n: int = Field(default=0)
    m: list[Any] = Field(default_factory=list)
    k: list[Any] = Field(default_factory=list)


class EDASummary(BaseModel):
    column: str
    dtype: str
    non_null_count: int
    null_ratio: float
    notes: str = ""
