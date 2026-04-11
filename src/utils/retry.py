"""Утилита для retry при сетевых ошибках GigaChat."""

from __future__ import annotations

import logging
import time
from typing import Any, Callable, TypeVar

import httpx
import httpcore

logger = logging.getLogger(__name__)

T = TypeVar("T")

_NETWORK_ERRORS = (
    httpx.ConnectError,
    httpx.RemoteProtocolError,
    httpx.ReadError,
    httpx.WriteError,
    httpx.TimeoutException,
    httpcore.ConnectError,
    ConnectionResetError,
    OSError,
)


def with_retry(
    func: Callable[..., T],
    *args: Any,
    retries: int = 3,
    delay: float = 5.0,
    backoff: float = 2.0,
    **kwargs: Any,
) -> T:
    """Вызывает func(*args, **kwargs) с повторами при сетевых ошибках.

    Args:
        func: вызываемая функция.
        retries: максимальное число попыток после первой ошибки.
        delay: задержка перед первым повтором (секунды).
        backoff: множитель для экспоненциальной задержки.
    """
    last_exc: Exception | None = None
    wait = delay
    for attempt in range(retries + 1):
        try:
            return func(*args, **kwargs)
        except _NETWORK_ERRORS as exc:
            last_exc = exc
            if attempt < retries:
                logger.warning(
                    "Сетевая ошибка (попытка %d/%d): %s. Повтор через %.1f сек.",
                    attempt + 1, retries + 1, exc, wait,
                )
                time.sleep(wait)
                wait *= backoff
            else:
                logger.error("Все %d попытки исчерпаны. Последняя ошибка: %s", retries + 1, exc)
    raise last_exc  # type: ignore[misc]
