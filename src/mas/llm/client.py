from __future__ import annotations

from typing import TYPE_CHECKING

from mas.settings import Settings

if TYPE_CHECKING:
    from langchain_gigachat import GigaChat


def build_gigachat(settings: Settings, *, request_timeout: float) -> "GigaChat":
    try:
        from langchain_gigachat import GigaChat
    except ImportError as e:
        raise ImportError("Установите langchain-gigachat (см. pyproject.toml)") from e

    return GigaChat(model=settings.gigachat_model, timeout=request_timeout)
