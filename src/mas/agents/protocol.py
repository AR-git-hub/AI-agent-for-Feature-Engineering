from __future__ import annotations

from typing import Protocol

from mas.domain.state import PipelineState


class LinearAgentStep(Protocol):
    def run(self, state: PipelineState, *, deadline: float) -> PipelineState: ...
