"""
Агент-генератор — придумывает идеи признаков на естественном языке.

Не пишет код, не оценивает качество.
На входе — отчёт аналитика (и опционально фидбек критика во втором раунде).
На выходе — список из до 5 описаний признаков.
"""

from __future__ import annotations

import json
import logging
from typing import Any

from langchain_core.messages import HumanMessage
from langchain_gigachat.chat_models import GigaChat

from src.utils.retry import with_retry

logger = logging.getLogger(__name__)

GENERATOR_SYSTEM_PROMPT = """
Ты — генератор признаков. Придумай 5 признаков для бинарной классификации.
НЕ пиши код — только описание логики на естественном языке.

## Вход

- Отчёт аналитика (схема данных, связи, проблемы).
- (Раунд 2) Фидбек критика — какие признаки слабые и почему.

## Что хорошо работает с CatBoost

- Агрегации по группам: mean/sum/count/std числового поля по категории
- Отношения и разности между числовыми полями
- Частотное кодирование категорий (доля значения в train)
- Target encoding (средний таргет по категории, считать ТОЛЬКО на train)
- Флаги: is_null, is_above_median, has_history
- Нормализация относительно группы: (значение - среднее_группы) / std_группы

## Чего избегать

- Признаки из колонок с >50% пропусков
- Константы (nunique=1)
- Копии существующих колонок без преобразования
- Leakage: колонки, кодирующие таргет или содержащие будущую информацию

## Раунд 2

Не повторяй слабые признаки. Используй другие таблицы, агрегации, взаимодействия.

## Ответ — ТОЛЬКО JSON-список из 5 элементов:

[
  {
    "name": "имя_признака",
    "description": "логика вычисления",
    "tables": ["таблицы"],
    "columns": ["колонки"],
    "hypothesis": "почему полезен для таргета"
  }
]
"""

def run_generator(
    llm: GigaChat,
    analyst_report: dict[str, Any],
    critic_feedback: dict[str, Any] | None = None,
    round_num: int = 1,
) -> list[dict[str, Any]]:
    """Запускает агента-генератора и возвращает список описаний признаков.

    Args:
        llm: инициализированный GigaChat.
        analyst_report: отчёт аналитика.
        critic_feedback: фидбек критика (только для раунда 2).
        round_num: номер раунда (1 или 2).

    Returns:
        Список dict с описаниями признаков (до 5 штук).
    """
    logger.info("[generator] Запуск раунд=%d", round_num)

    prompt_parts = [
        f"## Отчёт аналитика\n```json\n{json.dumps(analyst_report, ensure_ascii=False, indent=2)}\n```"
    ]

    if critic_feedback and round_num > 1:
        prompt_parts.append(
            f"\n## Фидбек критика (раунд {round_num - 1})\n"
            f"```json\n{json.dumps(critic_feedback, ensure_ascii=False, indent=2)}\n```\n"
            "Учти фидбек: не повторяй слабые признаки, предложи другие подходы."
        )

    prompt_parts.append(
        "\nСгенерируй ровно 5 новых признаков. Верни ТОЛЬКО JSON-список."
    )

    user_message = "\n".join(prompt_parts)

    messages = [
        {"role": "system", "content": GENERATOR_SYSTEM_PROMPT},
        {"role": "user", "content": user_message},
    ]

    logger.info("[generator] Отправка запроса в GigaChat (раунд %d)...", round_num)
    response = with_retry(llm.invoke, messages)
    raw = response.content.strip()
    logger.info("[generator] Ответ получен, длина=%d символов", len(raw))

    # Парсим JSON из ответа
    features = _parse_features(raw)
    logger.info("[generator] Распознано признаков: %d", len(features))
    for i, f in enumerate(features, 1):
        logger.info("[generator]   %d. %s — %s", i, f.get("name", "?"), f.get("hypothesis", "")[:80])

    return features


def _parse_features(raw: str) -> list[dict[str, Any]]:
    """Извлекает JSON-список признаков из ответа LLM."""
    # Убираем markdown-блоки если есть
    text = raw
    if "```json" in text:
        text = text.split("```json", 1)[1]
        text = text.split("```", 1)[0]
    elif "```" in text:
        text = text.split("```", 1)[1]
        text = text.split("```", 1)[0]

    text = text.strip()

    try:
        result = json.loads(text)
        if isinstance(result, list):
            return result[:5]
        if isinstance(result, dict) and "features" in result:
            return result["features"][:5]
    except json.JSONDecodeError:
        logger.warning("[generator] Не удалось распарсить JSON, пробуем найти список в тексте")

    # Последняя попытка — найти [...] в тексте
    start = raw.find("[")
    end = raw.rfind("]")
    if start != -1 and end != -1:
        try:
            return json.loads(raw[start:end + 1])[:5]
        except json.JSONDecodeError:
            pass

    logger.error("[generator] Не удалось извлечь признаки из ответа LLM")
    return []
