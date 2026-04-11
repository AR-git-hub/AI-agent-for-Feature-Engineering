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

from src.utils.retry import with_retry

logger = logging.getLogger(__name__)

CRITIC_SYSTEM_PROMPT = """
Ты — критик качества признаков. Получаешь статистику по набору признаков и выносишь вердикт.

Задача: объективно оценить каждый признак по метрикам и сформировать КОНКРЕТНЫЙ actionable фидбек
для генератора — что заменить и чем.

---

## Пороги оценки признаков

### Сила признака (насколько информативен для таргета)
- MI >= 0.05 и |pearson| >= 0.10 → СИЛЬНЫЙ  (score 0.80–1.00, verdict="strong")
- MI >= 0.01 или |pearson| >= 0.05 → ПОЛЕЗНЫЙ (score 0.50–0.79, verdict="useful")
- MI <  0.01 и |pearson| <  0.05 → СЛАБЫЙ   (score 0.20–0.49, verdict="weak")

### Признак-мусор → verdict="weak", score <= 0.2
- nunique = 1 — константа
- null_pct > 80% — почти весь NaN
- high_collinearity = True (max_corr_with_others > 0.85) — дублирует другой признак

### Подозрение на leakage → verdict="suspicious_leakage", score <= 0.2
- |pearson| > 0.9
- mutual_info > 0.5

---

## Правило overall_score (0.0–1.0)

1. Отбрось признаки с verdict="suspicious_leakage" (даже один такой — серьёзный штраф).
2. Для остальных: base = среднее значение score.
3. Штраф −0.10 за каждую пару с high_collinearity.
4. Штраф −0.05 за каждый признак с null_pct > 30%.
5. Если хотя бы один leakage — штраф −0.30 к итогу.
6. Итог ограничь диапазоном [0.0, 1.0].

Интерпретация:
- overall_score >= 0.75 → набор силён, need_second_round=false
- 0.55 <= overall_score < 0.75 → приемлем, need_second_round=true (если есть время)
- overall_score < 0.55 → слабый, need_second_round=true

---

## Правила feedback_for_generator

Фидбек должен быть конкретным и actionable:
- НЕ пиши "признаки слабые" — пиши что именно заменить и чем
- Укажи конкретные таблицы/колонки которые стоит использовать вместо слабых
- Если есть коллинеарность — скажи какой из пары оставить (с большим MI)
- Если все признаки из одной таблицы — предложи добавить признаки из других таблиц
- Если признаки слишком простые — предложи более сложные агрегации или взаимодействия

Пример хорошего фидбека:
"day_of_week_encoded слабый (MI=0.005) — заменить на user_reorder_rate из order_items.
order_frequency и basket_to_order_ratio коллинеарны (corr=0.91) — оставить только basket_to_order_ratio.
Все признаки на уровне пользователя — добавить признак на уровне пары (user_id, product_id):
частота покупки конкретного товара этим пользователем из order_items."

---

## Ответ — ТОЛЬКО JSON, без пояснений до или после:

{
  "ranking": ["лучший_признак", "второй", "третий", "четвертый", "худший"],
  "scores": {
    "имя_признака": {
      "score": 0.75,
      "verdict": "strong",
      "reason": "конкретное обоснование: MI=0.04, pearson=0.15, нет пропусков"
    }
  },
  "overall_score": 0.65,
  "need_second_round": true,
  "need_clarification": false,
  "clarification_question": "",
  "feedback_for_generator": "конкретный actionable фидбек: что заменить, чем, из каких таблиц",
  "confidence": "high"
}

Допустимые значения verdict: "strong" | "useful" | "weak" | "suspicious_leakage"
Допустимые значения confidence: "high" | "medium" | "low"
"""

CRITIC_COMPARE_PROMPT = """
Ты — критик. Сравни два набора признаков и выбери лучший.

## Критерии сравнения (по убыванию приоритета)

1. LEAKAGE — дисквалифицирует набор:
   |pearson| > 0.9 или MI > 0.5 для любого признака → этот набор проигрывает автоматически.

2. СУММАРНЫЙ MI — чем выше сумма MI по всем признакам без leakage, тем лучше.

3. РАЗНООБРАЗИЕ — штраф за коллинеарность:
   если max_corr_with_others > 0.85 для нескольких пар — набор хуже.

4. КАЧЕСТВО ПОКРЫТИЯ — штраф за пропуски:
   средний null_pct по признакам должен быть минимальным.

5. При равенстве по MI — предпочти набор с меньшим числом слабых признаков (MI < 0.01).

## Ответ — ТОЛЬКО JSON:

{
  "winner": 1,
  "reason": "конкретное обоснование: суммарный MI раунда 2 = 0.18 vs 0.12 у раунда 1; нет leakage в обоих",
  "round1_overall": 0.65,
  "round2_overall": 0.72
}
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
    response = with_retry(llm.invoke, messages)
    raw = (response.content if isinstance(response.content, str) else str(response.content)).strip()
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

    response = with_retry(llm.invoke, messages)
    raw_compare = (response.content if isinstance(response.content, str) else str(response.content)).strip()
    result = _parse_json_response(raw_compare)

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
