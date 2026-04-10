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

CRITIC_SYSTEM_PROMPT = """Ты — агент-критик в системе генерации признаков для бинарной классификации.

Твоя задача — оценить набор признаков по метрикам и решить:
1. Какие признаки сильные, какие слабые.
2. Нужен ли второй раунд генерации.
3. Какой набор (раунд 1 или раунд 2) лучше (если сравниваем два набора).

## Метрики, которые ты получаешь (по каждому признаку)

- `pearson` — корреляция Пирсона с таргетом (|pearson| > 0.05 — хорошо)
- `spearman` — ранговая корреляция (устойчива к выбросам)
- `mutual_info` — mutual information (MI > 0.01 — хорошо, > 0.05 — отлично)
- `null_pct` — процент пропусков (> 50% — плохо, > 80% — очень плохо)
- `nunique` — число уникальных значений (1 = константа = мусор)
- `max_corr_with_others` — максимальная корреляция с другими признаками (> 0.85 — дублирует)
- `high_collinearity` — флаг высокой мультиколлинеарности

## Подозрение на leakage

Если |pearson| > 0.9 или MI > 0.5 — признак подозрительно хорош. Возможен data leakage.

## Формат ответа

Верни ТОЛЬКО валидный JSON без лишнего текста:

```json
{
  "ranking": ["лучший_признак", "второй", ...],
  "scores": {
    "имя_признака": {
      "score": 0.75,
      "verdict": "strong | weak | suspicious_leakage",
      "reason": "объяснение на русском"
    }
  },
  "overall_score": 0.65,
  "need_second_round": true,
  "need_clarification": false,
  "clarification_question": "",
  "feedback_for_generator": "что улучшить в следующем раунде",
  "confidence": "high | medium | low"
}
```

## Критерии overall_score

- 0.8+ — отличный набор, второй раунд не нужен
- 0.6–0.8 — хороший набор, второй раунд желателен
- < 0.6 — слабый набор, второй раунд обязателен
"""

CRITIC_COMPARE_PROMPT = """Ты — агент-критик. Сравни два набора признаков и выбери лучший.

Лучший набор — тот, где признаки в совокупности дают:
- максимальное покрытие информации (сумма MI)
- минимальную мультиколлинеарность
- минимум пропусков
- без подозрений на leakage

Верни ТОЛЬКО валидный JSON:

```json
{
  "winner": 1,
  "reason": "объяснение выбора",
  "round1_overall": 0.65,
  "round2_overall": 0.72
}
```
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
