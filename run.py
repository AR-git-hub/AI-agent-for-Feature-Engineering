"""Точка входа: `python run.py` (ТЗ). Линейный MAS без отдельного оркестратора."""
from __future__ import annotations

import logging
import os

from dotenv import load_dotenv

load_dotenv()

from src.mas.logging_setup import configure_logging

configure_logging()

logger = logging.getLogger("run")


def run_pipeline():
    from src.mas.agents import (
        AnswerAgent,
        DatasetAgent,
        FeatureGenerationAgent,
        FeatureSelectionAgent,
    )
    from src.mas.config import CONFIGS_DIR, DATA_DIR, OUTPUT_DIR
    from src.mas.context import RunContext

    _ = os.environ.get("GIGACHAT_CREDENTIALS"), os.environ.get("GIGACHAT_SCOPE")

    ctx = RunContext(data_dir=DATA_DIR, output_dir=OUTPUT_DIR, configs_dir=CONFIGS_DIR)

    steps = [
        ("DatasetAgent", DatasetAgent()),
        ("FeatureGenerationAgent", FeatureGenerationAgent()),
        ("FeatureSelectionAgent", FeatureSelectionAgent()),
        ("AnswerAgent", AnswerAgent()),
    ]
    for name, agent in steps:
        logger.info(">>> %s: start", name)
        ctx = agent.run(ctx)
        logger.info("<<< %s: done", name)

    return ctx


if __name__ == "__main__":
    run_pipeline()
