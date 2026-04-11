# SBER MAS — сабмит

Гайд для команды: **[docs/TEAM_GUIDE.md](docs/TEAM_GUIDE.md)**.

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



# Гайд по репозиторию для команды

Единый документ: цель проекта, структура, как запускать, кто за что отвечает, контракты между агентами и как не сломать сабмит.

---

## 1. Цель и ограничения платформы

**Задача:** линейный MAS на Python + GigaChat (`langchain-gigachat`): читает вход из `data/`, генерирует **1–5** новых признаков для бинарной классификации, пишет **`output/train.csv`** и **`output/test.csv`** со **всеми исходными колонками** плюс новые (одинаковый набор и **порядок** новых колонок в train и test).

**Жёсткие правила чекера** (`src/utils/check_submission.py`):

- Зависимости только из `pyproject.toml` (в т.ч. маркеры: `pandas`, `numpy`, `catboost`, `langchain-gigachat`, `python-dotenv`).
- Запуск решения: **`python run.py`** из корня репо.
- Вход: только **`data/train.csv`**, **`data/test.csv`**, **`data/readme.txt`**.
- Выход: только **`output/train.csv`**, **`output/test.csv`**.
- В корне должен быть **`.env`** с строками `GIGACHAT_CREDENTIALS` и `GIGACHAT_SCOPE`.
- Лимит времени прогона чекера: **600 с**.

Локальная проверка перед сабмитом:

```bash
python src/utils/check_submission.py
```

---

## 2. Структура репозитория

| Путь | Назначение |
|------|------------|
| `run.py` | Единственная точка входа: `chdir` в корень, `load_dotenv(.env)`, при необходимости добавляет `src/` в `sys.path`, запускает пайплайн. |
| `pyproject.toml` | Зависимости и setuptools: пакет `mas` из каталога `src/`. |
| `uv.lock` | Зафиксированные версии под `uv sync` (если ведёте lock). |
| `.env` / `.env.example` | Креды GigaChat и scope; для оффлайна токен может быть пустым. |
| `data/` | **Только вход** организаторов / локального теста. Не писать сюда из кода пайплайна. |
| `output/` | **Только выход** пайплайна. В `.gitignore` — не коммитить артефакты прогонов. |
| `src/mas/` | Основной код: домен, сервисы, агенты, LLM, пайплайн. |
| `src/utils/` | `check_submission.py` (валидация), `baseline.py` (пример), `scoring.py` (логика платформы, **локально не импортируется** чекером). |
| `docs/` | Документация команды (этот файл). |

**Корень репо в коде:** `mas.paths.repo_root()` — от `__file__`, не от настроек IDE.

---

## 3. Окружение и запуск

### Вариант A: `uv` (как у организаторов)

```bash
uv venv
uv sync
python run.py
```

### Вариант B: `pip`

```bash
python -m venv .venv
.venv\Scripts\activate   # Windows
pip install -e .
python run.py
```

Перед сабмитом положите реальные входные файлы в `data/` и выполните:

```bash
python src/utils/check_submission.py
```

---

## 4. Пайплайн (линейный MAS)

Порядок фиксирован в `src/mas/pipeline/submission.py`:

```text
Analyst (1) → FeatureEngineer (2) → Selector (3) → Exporter (4)
```

Каждый шаг: `agent.run(state, deadline=...) -> PipelineState`.

### Общие правила для всех агентов

- **Не читать** ничего вне `data/` (кроме кода и `.env`).
- **Не писать** в `data/`; **писать в `output/`** только агент 4 (или общий пост-процессор, если вы его введёте — тогда явно зафиксировать в гайде).
- Уважать **`deadline`** (монотонные часы): тяжёлые циклы и вызовы LLM ограничивать (`MAS_PIPELINE_BUDGET_SEC`, таймаут на запрос в `build_gigachat`).
- При пустом `GIGACHAT_CREDENTIALS` сейчас используется **оффлайн-ветка** (эвристики без API) — удобно для CI/чекера без секретов.

---

## 5. Контракт `PipelineState` (`src/mas/domain/state.py`)

Поле | Кто заполняет | Смысл
------|----------------|------
`repo_root` | Оркестратор | Корень репо.
`readme_path`, `readme_text` | Агент 1 | `data/readme.txt`.
`id_column`, `target_column` | Агент 1 | Схема данных (таргет только в train).
`train_in`, `test_in` | Агент 1 | Сырые датафреймы из CSV.
`eda`, `feature_configs` | Агент 1 (и пересчёт после агента 2) | EDA по колонкам; `(n,m,k)` пока заглушки.
`candidate_feature_ids` | Агент 1 | Список **id** кандидатов в каталоге (`zscore:col`, `mul:a:b`, `synthetic:const`).
`train_enriched`, `test_enriched` | Агент 2 | Исходные колонки + сгенерированные.
`new_feature_names` | Агент 2 | Имена **новых** колонок (не из исходного train).
`top_feature_names` | Агент 3 | Подмножество новых имён, **1–5** штук, один порядок для train/test.
`transcripts` | Все | Короткие текстовые логи/обоснования по ключам `agent1`…`agent4`.

Любой PR, меняющий поля состояния, обновляет этот раздел и всех потребителей.

---

## 6. Роли по агентам (кто что делает)

### Агент 1 — Аналитик (`src/mas/agents/analyst.py`)

**Ответственность:** загрузка `data/*`, определение `id`/`target`, базовый EDA и заглушки конфигов, построение **`candidate_feature_ids`** для агента 2.

**Сервисы:** `mas.services.io`, `preprocessing`, `eda`, `feature_config`, `feature_catalog.build_catalog_ids`.

**Зона для тулов (сокомандник):** новые атомарные функции / LangChain `@tool` — лучше вынести в отдельный модуль (например `src/mas/tools/analyst.py`), не раздувать `analyst.py`. Тулы только читают из `data/`, не пишут в `output/`.

### Агент 2 — Конструктор признаков (`feature_engineer.py`)

**Ответственность:** по readme + EDA + конфигам выбрать id из `candidate_feature_ids`, вычислить признаки на train и test (`apply_catalog_features` в `feature_catalog.py`), обновить `train_enriched` / `test_enriched`, пересчитать EDA/конфиги.

**Согласование с агентом 1:** формат строк в `candidate_feature_ids` и парсер в `apply_catalog_features` — одна точка правды; менять синхронно.

### Агент 3 — Селектор (`selector.py`)

**Ответственность:** из `new_feature_names` выбрать **1–5** имён для финального сабмита (одинаковый порядок в train и test).

### Агент 4 — Экспортёр (`exporter.py`)

**Ответственность:** собрать `train_in`/`test_in` + колонки из `top_feature_names` из обогащённых таблиц, санация NaN на тесте, запись **`output/train.csv`** и **`output/test.csv`**.

---

## 7. LLM и секреты

- Клиент: `src/mas/llm/client.py` → `GigaChat` из `langchain-gigachat`, модель по умолчанию `GigaChat-2-Max` (`GIGACHAT_MODEL` в `.env` опционально).
- Переменные: `GIGACHAT_CREDENTIALS`, `GIGACHAT_SCOPE` (для соревнования часто `GIGACHAT_API_CORP`).
- **Не коммить** в публичный репозиторий реальные токены. Для приватного репо — политика команды сама; для соревнования — отдельный zip с `.env` по правилам организаторов.

---

## 8. Git и GitHub

- Ветка разработки по договорённости: **`main`**.
- Перед merge: `python src/utils/check_submission.py` (с положенными в `data/` файлами).
- Проблемы с `schannel` / TLS при `git push` по HTTPS: см. обсуждение в команде — типичный обход **SSH remote** (`git@github.com:ORG/REPO.git`).

---

## 9. Чеклист перед merge в `main`

1. `python run.py` без ошибок.
2. `python src/utils/check_submission.py` — **OK**.
3. Нет записи в `data/` из кода; `output/` не коммитим (или чистим осознанно).
4. Обновлены `docs/TEAM_GUIDE.md` / README, если менялись контракты или команды.

---

## 10. Куда писать с вопросами

- **Контракт агентов / состояние / сабмит** — тимлид / владелец интеграции пайплайна.
- **Тулы агента 1** — ответственный за `mas.tools` + ревью со стороны агента 2 на совместимость `candidate_feature_ids`.
- **Каталог и генерация признаков** — агенты 1–2 согласуют формат id в `feature_catalog.py`.
