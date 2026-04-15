"""Точка входа: python run.py запускает полный мультиагентный пайплайн."""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from langchain_gigachat.chat_models import GigaChat

from src.agents.orchestrator import run_pipeline

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

DATA_DIR = Path("data")
OUTPUT_DIR = Path("output")


def build_gigachat(config: dict[str, Any] | None = None) -> GigaChat:
    gc_cfg = (config or {}).get("gigachat", {})
    credentials = os.getenv("GIGACHAT_CREDENTIALS")
    scope = os.getenv("GIGACHAT_SCOPE")
    if not credentials:
        raise RuntimeError("Missing GIGACHAT_CREDENTIALS in environment")
    if not scope:
        raise RuntimeError("Missing GIGACHAT_SCOPE in environment")

    return GigaChat(
        credentials=credentials,
        scope=scope,
        model=gc_cfg.get("model", "GigaChat-2-Max"),
        temperature=float(gc_cfg.get("temperature", 0.2)),
        timeout=int(gc_cfg.get("timeout", 120)),
        verify_ssl_certs=bool(gc_cfg.get("verify_ssl_certs", False)),
    )


def main() -> None:
    load_dotenv()
    logger.info("Инициализация GigaChat...")
    llm = build_gigachat()
    logger.info("GigaChat готов. Запуск оркестратора.")
    run_pipeline(llm)


if __name__ == "__main__":
    main()
