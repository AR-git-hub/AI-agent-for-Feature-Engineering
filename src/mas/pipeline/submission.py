from __future__ import annotations

from mas.agents.analyst import AnalystAgent
from mas.agents.exporter import ExporterAgent
from mas.agents.feature_engineer import FeatureEngineerAgent
from mas.agents.protocol import LinearAgentStep
from mas.agents.selector import SelectorAgent
from mas.domain.state import PipelineState
from mas.paths import repo_root
from mas.settings import Settings


def main() -> int:
    settings = Settings()
    deadline = settings.monotonic_deadline()
    root = repo_root()

    state = PipelineState(repo_root=root)
    steps: list[LinearAgentStep] = [
        AnalystAgent(settings),
        FeatureEngineerAgent(settings),
        SelectorAgent(settings),
        ExporterAgent(settings),
    ]
    for agent in steps:
        state = agent.run(state, deadline=deadline)
    return 0
