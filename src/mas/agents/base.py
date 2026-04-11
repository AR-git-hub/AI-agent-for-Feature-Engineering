from __future__ import annotations

import os


def clip_text(text: str, max_chars: int = 8000) -> str:
    if len(text) <= max_chars:
        return text
    return text[:max_chars] + "\n...[truncated]..."


def gigachat_configured() -> bool:
    return bool(os.environ.get("GIGACHAT_CREDENTIALS", "").strip())
