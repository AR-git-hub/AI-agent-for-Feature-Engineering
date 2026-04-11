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

from langchain_gigachat.chat_models import GigaChat

from src.utils.retry import with_retry

logger = logging.getLogger(__name__)

GENERATOR_SYSTEM_PROMPT = """
Ты — генератор признаков для задачи бинарной классификации с последующим CatBoost.
Твоя задача — придумать РОВНО 5 СИЛЬНЫХ признаков на основе отчёта аналитика.

НЕ пиши код. Только формулируй чёткое ТЗ на естественном языке.
Кодер реализует код по твоему описанию.

---

## Что получишь на вход

- Отчёт аналитика: точные имена таблиц, колонок (numeric_columns, categorical_columns,
  datetime_columns), связи (joins), рекомендации (join_recommendations), проблемы, leakage_risks.
- (Раунд 2) Фидбек критика: какие признаки оказались слабыми и что исправить.

---

## Главные правила

1. **Используй ТОЛЬКО те имена колонок и таблиц, которые есть в отчёте аналитика.**
   Никогда не выдумывай `user_id`, `orders.csv` и т.п., если их нет в отчёте.

2. **Читай внимательно tables[*].columns_sample и поля numeric_columns/categorical_columns.**
   Если колонка в categorical_columns — она СТРОКОВАЯ (например 'may','mon','admin').
   Для неё НИКОГДА не предлагай sin/cos/log и арифметику — только freq_encoding/target_encoding
   или явный маппинг (например month → {'jan':1,...,'dec':12}).

3. **Каждый признак ДОЛЖЕН быть реализуем из доступных таблиц.** Если в train только [id,target],
   все признаки строятся через join с доп.таблицами по ключу.

4. **Никакого data leakage.** Избегай колонок из leakage_risks. target encoding считается
   ТОЛЬКО на train и мапится на test.

5. **Избегай колонок с >50% пропусков**, констант (nunique=1), прямых копий существующих колонок.

6. **5 разных признаков**: не дублируй идеи с маленькими вариациями. Разнообразие > микро-оптимизации.

---

## Каталог типов признаков

### A. Для категориальных колонок
- **frequency_encoding**: доля встречаемости значения в train (map)
- **target_encoding**: средний таргет по значению (считать только на train!)
- **ordinal mapping**: для упорядоченных категорий (month, education level) — явный dict

### B. Для числовых колонок (из доп.таблиц 1:1)
- Прямой map значения по ключу
- Отношения/разности пар числовых колонок: `a / (b + 1e-9)`, `a - b`
- Нормализация по категории: `(x - mean_x_by_group) / (std_x_by_group + 1e-9)`
- Бининг: квантильные бакеты, флаг is_above_median
- Флаги: is_zero, is_missing, равенство специальному значению (например pdays==999)

### C. Для временных/порядковых колонок
- Циклические синусы/косинусы — ТОЛЬКО ПОСЛЕ численного маппинга (не напрямую над строкой!)
- Разности дат (days between)
- Бакеты: неделя/месяц/квартал

### D. Для связей 1:N (если есть таблица событий)
Сначала groupby + agg, затем map/merge по ключу:
- count(*), sum, mean, std, max, min, nunique
- Доля конкретного значения: `count(col=='X') / count(*)`
- Давность: `today - max(date)`

### E. Комбинации
- Взаимодействия пар категорий (concat строк + target encoding)
- Счётчики одновременной встречаемости

---

## Раунд 2 — улучшение

1. Не повторяй слабые признаки (даже в вариациях).
2. Используй другие колонки / другие типы агрегаций из того же отчёта.
3. Если в раунде 1 были только числовые — попробуй категориальные через target encoding.
4. Проверяй, что новый признак не коллинеарен с предыдущими сильными.

---

## Формат ответа — ТОЛЬКО JSON-массив из ровно 5 объектов

Без текста до/после, без markdown.

[
  {
    "name": "snake_case_name",
    "description": "Пошагово: из какой таблицы взять какие колонки, каким ключом джойнить, какая формула/агрегация, как обработать NaN",
    "tables": ["точное_имя_таблицы.csv"],
    "join_key": "точное_имя_ключа (часто совпадает с id_column)",
    "columns": ["точные_имена_колонок"],
    "encoding": "numeric_map | freq_encoding | target_encoding | ordinal_map | ratio | groupby_agg",
    "hypothesis": "почему этот признак должен коррелировать с таргетом"
  }
]

## Пример (для банковских данных с колонками age, job, month из client_data.csv):

{
  "name": "job_target_encoding",
  "description": "Средний таргет по значению колонки 'job'. Соединить train с client_data по client_id, сгруппировать по job и посчитать среднее target на train. Для test — использовать эту же таблицу соответствий. NaN заполнить глобальным средним target по train.",
  "tables": ["client_data.csv"],
  "join_key": "client_id",
  "columns": ["job"],
  "encoding": "target_encoding",
  "hypothesis": "Разные профессии имеют разную склонность оформлять депозит (например пенсионеры vs студенты)"
}
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
    raw = (response.content if isinstance(response.content, str) else str(response.content)).strip()
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
