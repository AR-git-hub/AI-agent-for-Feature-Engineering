"""
Агент-критик — оценивает качество набора признаков по прокси-метрикам.

Не обучает модель. Работает с таблицей метрик от кодера (compute_stats).
Возвращает ранжирование, оценки и фидбек для генератора.
"""

from __future__ import annotations

import json
import logging
from typing import Any

from langchain_gigachat.chat_models import GigaChat

logger = logging.getLogger(__name__)

CRITIC_SYSTEM_PROMPT = """
Ты — критик. Оцени набор признаков по метрикам и верни JSON.

## Пороги оценки

Сила: |pearson|>0.05 или MI>0.01 — полезен. MI>0.05 — сильный.
Мусор: nunique=1, null_pct>80%, zeros_pct>95%.
Дубль: max_corr_with_others>0.85 — дублирует другой признак.
Leakage: |pearson|>0.9 или MI>0.5 — подозрителен.

## overall_score

>=0.8 — второй раунд не нужен. 0.6–0.8 — желателен. <0.6 — обязателен.

## Ответ — ТОЛЬКО JSON:

{
  "ranking": ["лучший", "второй", ...],
  "scores": {
    "имя": {"score": 0.75, "verdict": "strong|weak|suspicious_leakage", "reason": "почему"}
  },
  "overall_score": 0.65,
  "need_second_round": true,
  "need_clarification": false,
  "clarification_question": "",
  "feedback_for_generator": "что улучшить",
  "confidence": "high|medium|low"
}
"""

CRITIC_COMPARE_PROMPT = """
Сравни два набора признаков. Лучший — максимум суммы MI
при минимуме корреляций между признаками, минимуме пропусков,
без leakage. Ответ — ТОЛЬКО JSON:

{"winner": 1, "reason": "почему", "round1_overall": 0.65, "round2_overall": 0.72}
"""

CRITIC_COMPARE_PROMPT = """
Сравни два набора признаков. Критерии (по приоритету):
1. Нет leakage (|pearson|>0.9 или MI>0.5 — дисквалификация признака)
2. Максимум суммарного MI оставшихся признаков
3. Все попарные корреляции между признаками < 0.85
4. Минимум пропусков

Ответ — ТОЛЬКО JSON:
{"winner": 1, "reason": "почему", "round1_overall": 0.65, "round2_overall": 0.72}
"""


def run_critic(
    llm: GigaChat,
    stats: dict[str, Any],
    round_num: int = 1,
) -> dict[str, Any]:
    """Оценивает набор признаков по метрикам.

    Args:
        llm: инициализированный GigaChat.
        stats: результат compute_stats (dict с ключом "features").
        round_num: номер раунда.

    Returns:
        dict с оценками, ранжированием и фидбеком.
    """
    logger.info("[critic] Запуск оценки (раунд %d), признаков=%d",
                round_num, len(stats.get("features", {})))

    user_message = (
        f"## Метрики признаков (раунд {round_num})\n"
        f"```json\n{json.dumps(stats, ensure_ascii=False, indent=2)}\n```\n\n"
        "Оцени набор и верни JSON."
    )

    messages = [
        {"role": "system", "content": CRITIC_SYSTEM_PROMPT},
        {"role": "user", "content": user_message},
    ]

    logger.info("[critic] Отправка запроса в GigaChat...")
    response = llm.invoke(messages)
    raw = response.content.strip()
    logger.info("[critic] Ответ получен, длина=%d символов", len(raw))

    result = _parse_json_response(raw)

    overall = result.get("overall_score", 0.0)
    need_r2 = result.get("need_second_round", False)
    confidence = result.get("confidence", "unknown")
    logger.info(
        "[critic] Оценка набора: overall_score=%.2f, need_second_round=%s, confidence=%s",
        overall, need_r2, confidence,
    )

    ranking = result.get("ranking", [])
    scores = result.get("scores", {})
    for name in ranking:
        s = scores.get(name, {})
        logger.info(
            "[critic]   %s: score=%.2f, verdict=%s — %s",
            name, s.get("score", 0.0), s.get("verdict", "?"), s.get("reason", "")[:60],
        )

    if result.get("need_clarification"):
        logger.info("[critic] Запрос уточнения у аналитика: %s",
                    result.get("clarification_question", ""))

    return result


def compare_rounds(
    llm: GigaChat,
    stats_round1: dict[str, Any],
    stats_round2: dict[str, Any],
    eval_round1: dict[str, Any],
    eval_round2: dict[str, Any],
) -> int:
    """Сравнивает два набора признаков и возвращает номер лучшего раунда (1 или 2).

    Args:
        llm: инициализированный GigaChat.
        stats_round1/2: метрики от compute_stats.
        eval_round1/2: оценки критика за каждый раунд.

    Returns:
        1 или 2 — номер лучшего раунда.
    """
    logger.info("[critic] Сравнение раундов 1 и 2...")

    user_message = (
        "## Раунд 1\n"
        f"Метрики:\n```json\n{json.dumps(stats_round1, ensure_ascii=False, indent=2)}\n```\n"
        f"Оценка:\n```json\n{json.dumps(eval_round1, ensure_ascii=False, indent=2)}\n```\n\n"
        "## Раунд 2\n"
        f"Метрики:\n```json\n{json.dumps(stats_round2, ensure_ascii=False, indent=2)}\n```\n"
        f"Оценка:\n```json\n{json.dumps(eval_round2, ensure_ascii=False, indent=2)}\n```\n\n"
        "Выбери лучший набор. Верни JSON."
    )

    messages = [
        {"role": "system", "content": CRITIC_COMPARE_PROMPT},
        {"role": "user", "content": user_message},
    ]

    response = llm.invoke(messages)
    result = _parse_json_response(response.content.strip())

    winner = int(result.get("winner", 1))
    reason = result.get("reason", "")
    logger.info("[critic] Победитель: раунд %d — %s", winner, reason[:80])
    return winner


def _parse_json_response(raw: str) -> dict[str, Any]:
    """Извлекает JSON из ответа LLM."""
    text = raw
    if "```json" in text:
        text = text.split("```json", 1)[1].split("```", 1)[0]
    elif "```" in text:
        text = text.split("```", 1)[1].split("```", 1)[0]
    text = text.strip()

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Ищем {...} в тексте
    start = raw.find("{")
    end = raw.rfind("}")
    if start != -1 and end != -1:
        try:
            return json.loads(raw[start:end + 1])
        except json.JSONDecodeError:
            pass

    logger.error("[critic] Не удалось распарсить JSON из ответа критика")
    return {
        "ranking": [],
        "scores": {},
        "overall_score": 0.5,
        "need_second_round": False,
        "need_clarification": False,
        "clarification_question": "",
        "feedback_for_generator": "",
        "confidence": "low",
    }
