# SBER MAS — сабмит

## Окружение

```bash
uv venv
uv sync
```

Активация venv по платформе (при необходимости), затем:

```bash
python run.py
```

Проверка требований организаторов (после размещения входных данных в `data/`):

```bash
python src/utils/check_submission.py
```

## Вход / выход

- Вход только из **`data/`**: `train.csv`, `test.csv`, `readme.txt`.
- Выход только в **`output/`**: `train.csv`, `test.csv` (все исходные колонки + 1–5 новых).

## Конфигурация

Файл **`.env`** в корне (в zip-сабмите обязателен):

```
GIGACHAT_CREDENTIALS=
GIGACHAT_SCOPE=GIGACHAT_API_CORP
```

Пустой `GIGACHAT_CREDENTIALS` допустим для оффлайн-ветки (эвристики без вызова API); для продакшн-прогона укажите токен.

Лимит времени пайплайна задаётся `MAS_PIPELINE_BUDGET_SEC` (по умолчанию 580 с; внешний таймаут чекера — 600 с).

## Код

Пакет **`mas`** расположен в `src/mas/`. Точка входа — **`run.py`** в корне.
