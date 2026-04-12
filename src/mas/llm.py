"""GigaChat: клиент и вызовы с логированием ответа и размышлений (reasoning_content)."""
from __future__ import annotations

import logging
import os
import time
from typing import Any, Iterator, Union

from gigachat import GigaChat
from gigachat.models import Chat, ChatCompletion, ChatCompletionChunk

logger = logging.getLogger("mas.gigachat")


DEFAULT_MODEL = "GigaChat-2-Max"

# Retry-политика для сетевых ошибок (SSL EOF, connection reset и т.п.)
_RETRY_ATTEMPTS = 4
_RETRY_BASE_DELAY = 2.0   # секунды (удваивается при каждой попытке)


def gigachat_client(**kwargs: Any) -> GigaChat:
    """Credentials из `.env`; модель из `GIGACHAT_MODEL` (по умолчанию GigaChat-2-Max)."""
    base: dict[str, Any] = {
        "credentials": os.environ["GIGACHAT_CREDENTIALS"],
        "scope": os.environ.get("GIGACHAT_SCOPE", "GIGACHAT_API_CORP"),
        "model": os.environ.get("GIGACHAT_MODEL", DEFAULT_MODEL),
        "verify_ssl_certs": False,  # корпоративный прокси / CORP-среда
        "timeout": 120,             # явный таймаут, чтобы не ждать вечно
    }
    base.update(kwargs)
    return GigaChat(**base)


def _is_retryable(exc: BaseException) -> bool:
    """Возвращает True для сетевых/SSL ошибок, которые стоит повторить."""
    msg = str(exc).lower()
    retryable_fragments = (
        "eof occurred",
        "unexpected_eof",
        "connection reset",
        "connection aborted",
        "ssl",
        "broken pipe",
        "timed out",
        "read timeout",
        "remote end closed",
    )
    return any(frag in msg for frag in retryable_fragments)


def log_chat_completion(completion: ChatCompletion, label: str = "") -> None:
    """Пишет в лог размышления (`reasoning_content`) и итоговый текст ответа."""
    prefix = f"[{label}] " if label else ""
    for i, choice in enumerate(completion.choices):
        msg = choice.message
        reasoning = getattr(msg, "reasoning_content", None) or ""
        content = (msg.content or "").strip()
        if reasoning and str(reasoning).strip():
            logger.info("%schoice=%s reasoning:\n%s", prefix, i, reasoning)
        else:
            logger.info("%schoice=%s reasoning: (нет)", prefix, i)
        preview = content if len(content) <= 8000 else content[:8000] + "\n… [обрезано]"
        logger.info("%schoice=%s content:\n%s", prefix, i, preview or "(пусто)")
    try:
        logger.info("%susage: %s", prefix, completion.usage)
    except Exception:
        logger.info("%susage: (недоступно)", prefix)


def chat_logged(
    client: GigaChat,
    payload: Union[Chat, dict[str, Any], str],
    *,
    label: str = "",
) -> ChatCompletion:
    """
    Обертка над `client.chat`: то же API, плюс лог размышлений и ответа.
    Автоматически повторяет запрос при сетевых/SSL ошибках (_RETRY_ATTEMPTS раз).
    В агентах вызывайте `chat_logged`, а не `client.chat`, чтобы логи были единообразны.
    """
    last_exc: BaseException | None = None
    delay = _RETRY_BASE_DELAY
    for attempt in range(1, _RETRY_ATTEMPTS + 1):
        try:
            logger.info("[%s] GigaChat request (chat), попытка %d/%d", label or "gigachat", attempt, _RETRY_ATTEMPTS)
            completion = client.chat(payload)
            log_chat_completion(completion, label=label)
            return completion
        except Exception as exc:
            last_exc = exc
            if _is_retryable(exc) and attempt < _RETRY_ATTEMPTS:
                logger.warning(
                    "[%s] Сетевая ошибка (попытка %d/%d), повтор через %.0f с: %s",
                    label or "gigachat", attempt, _RETRY_ATTEMPTS, delay, exc,
                )
                time.sleep(delay)
                delay *= 2
            else:
                raise
    raise RuntimeError("Все попытки chat_logged исчерпаны") from last_exc


def stream_logged(
    client: GigaChat,
    payload: Union[Chat, dict[str, Any], str],
    *,
    label: str = "",
) -> Iterator[ChatCompletionChunk]:
    """
    Стриминг `client.stream`: собирает reasoning/content по чанкам и пишет один лог в конце.
    Для пошагового вывода в консоль добавьте свой цикл по yield из этой функции.
    """
    from collections import defaultdict

    buf_r: dict[int, list[str]] = defaultdict(list)
    buf_c: dict[int, list[str]] = defaultdict(list)
    logger.info("%sGigaChat stream start", label or "gigachat")
    for chunk in client.stream(payload):
        for ch in chunk.choices:
            d = ch.delta
            if d.reasoning_content:
                buf_r[ch.index].append(d.reasoning_content)
            if d.content:
                buf_c[ch.index].append(d.content)
        yield chunk
    for idx in sorted(set(buf_r) | set(buf_c)):
        r = "".join(buf_r.get(idx, [])).strip()
        c = "".join(buf_c.get(idx, [])).strip()
        logger.info("%sstream choice=%s reasoning:\n%s", label, idx, r or "(нет)")
        logger.info("%sstream choice=%s content:\n%s", label, idx, c or "(пусто)")
