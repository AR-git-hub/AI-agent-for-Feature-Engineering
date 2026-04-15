"""
Оркестратор — центральный агент, управляющий пайплайном.

Не работает с сырыми данными. Вызывает суб-агентов через инструменты,
следит за бюджетом времени, принимает стратегические решения.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any

import pandas as pd
from langchain_gigachat.chat_models import GigaChat

import numpy as np

from src.agents.analyst import run_analyst
from src.agents.coder import build_features, compute_stats
from src.agents.critic import run_critic
from src.agents.generator import run_generator

logger = logging.getLogger(__name__)

OUTPUT_DIR = Path("output")
DATA_DIR = Path("data")
TIME_BUDGET = 500          # секунд (оставляем буфер от лимита 600)
TIME_FOR_ROUND2 = 200      # минимум секунд для запуска раунда 2
TIME_EMERGENCY = 60        # меньше этого — сразу save

# ---------------------------------------------------------------------------
# Бюджеты на отдельные шаги (soft — проверяются post-hoc, не прерывают)
# Сумма ≈ TIME_BUDGET с буфером. Если шаг существенно превысил свой бюджет,
# оркестратор логирует ворнинг и пропускает следующие опциональные шаги.
# ---------------------------------------------------------------------------
STEP_BUDGETS = {
    "analyst":       80,   # 6-15 tool calls, каждый с LLM-раундом
    "generator":     45,   # один LLM-вызов, иногда ретраится
    "build_features":90,   # до 3 попыток с debug-циклом
    "compute_stats":  5,   # чистый numpy/sklearn, быстро
    "critic":        30,   # один LLM-вызов
    "save":          10,   # pandas to_csv
}


def save(
    df_train: pd.DataFrame,
    df_test: pd.DataFrame,
    analyst_report: dict[str, Any],
    feature_cols: list[str],
) -> None:
    """Сохраняет финальные признаки в output/train.csv и output/test.csv.

    Формат: все исходные колонки input/train.csv + признаки (не более 5).
    check_submission требует:
      - все оригинальные колонки присутствуют в выводе
      - набор признаков (имена и порядок) одинаков в train и test
    """
    id_col = analyst_report.get("id_column", "client_id")

    # Берём не более 5 признаков, которые есть в df_train
    feat_cols = [c for c in feature_cols[:5] if c in df_train.columns]
    logger.info("[save] Сохранение признаков: %s", feat_cols)

    # Читаем исходные файлы чтобы получить все оригинальные колонки
    src_train = pd.read_csv(DATA_DIR / "train.csv")
    src_test = pd.read_csv(DATA_DIR / "test.csv")

    # Присоединяем признаки train к исходному train по id_col
    out_train = src_train.merge(
        df_train[[id_col] + feat_cols],
        on=id_col,
        how="left",
    )

    # Для test берём только признаки, которые есть в df_test;
    # если какого-то признака нет — добавляем колонку с 0 чтобы наборы совпали
    test_feat_df = df_test[[c for c in [id_col] + feat_cols if c in df_test.columns]].copy()
    for col in feat_cols:
        if col not in test_feat_df.columns:
            test_feat_df[col] = 0.0
            logger.warning("[save] Признак '%s' отсутствует в df_test, заполнен нулями", col)
    test_feat_df = test_feat_df[[id_col] + feat_cols]

    out_test = src_test.merge(
        test_feat_df,
        on=id_col,
        how="left",
    )

    # Добиваем fallback-признаками если OOF-фильтрация отбросила часть
    if len(feat_cols) < 5:
        slots = 5 - len(feat_cols)
        logger.info("[save] Осталось %d признаков, добиваем %d fallback-признаками", len(feat_cols), slots)
        existing = set(out_train.columns)
        # Пробуем L1 (freq encoding), потом L2 (numeric map)
        for strategy_name, strategy in [
            ("L1_freq", _fallback_l1_freq_encoding),
            ("L2_num", _fallback_l2_numeric_map),
        ]:
            try:
                result = strategy(src_train, src_test, id_col)
            except Exception:
                continue
            if result is None:
                continue
            fb_train, fb_test, fb_feats = result
            for fb_col in fb_feats:
                if fb_col in existing or slots <= 0:
                    continue
                out_train[fb_col] = fb_train[fb_col].values
                out_test[fb_col] = fb_test[fb_col].values
                slots -= 1
                existing.add(fb_col)
                logger.info("[save] Добавлен fallback-признак: %s (%s)", fb_col, strategy_name)
            if slots <= 0:
                break

    # Гарантируем одинаковый порядок признаков в обоих файлах
    # (merge не меняет порядок колонок src, признаки добавляются в конец)
    OUTPUT_DIR.mkdir(exist_ok=True)
    out_train.to_csv(OUTPUT_DIR / "train.csv", index=False)
    out_test.to_csv(OUTPUT_DIR / "test.csv", index=False)

    logger.info("[save] output/train.csv: shape=%s, колонки=%s", out_train.shape, list(out_train.columns))
    logger.info("[save] output/test.csv:  shape=%s, колонки=%s", out_test.shape, list(out_test.columns))


def _get_feature_cols(df_train: pd.DataFrame, id_col: str, target_col: str) -> list[str]:
    return [c for c in df_train.columns if c not in (id_col, target_col)]


def _oof_quality_check(
    df_train: pd.DataFrame,
    target_col: str,
    feature_cols: list[str],
) -> list[str]:
    """Проверяет признаки через out-of-fold: считает MI/pearson на hold-out.

    Если признак на hold-out теряет >80% MI по сравнению с полным train,
    это признак leakage (target encoding без OOF). Такие фичи отбрасываются.

    Также отбрасывает фичи с MI < 0.001 на hold-out (бесполезные).

    Returns:
        Отфильтрованный список признаков (может быть короче).
    """
    from sklearn.feature_selection import mutual_info_classif
    from sklearn.model_selection import StratifiedKFold

    if not feature_cols or target_col not in df_train.columns:
        return feature_cols

    y = np.asarray(df_train[target_col].values)
    X = df_train[feature_cols].apply(pd.to_numeric, errors="coerce").fillna(0).values

    # MI на полном train
    try:
        mi_full = mutual_info_classif(X, y, random_state=42)
    except Exception:
        return feature_cols

    # MI на out-of-fold (один фолд в качестве hold-out для скорости)
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    _, val_idx = next(iter(skf.split(X, y)))
    try:
        mi_oof = mutual_info_classif(X[val_idx], y[val_idx], random_state=42)
    except Exception:
        return feature_cols

    good_cols: list[str] = []
    for i, col in enumerate(feature_cols):
        full_mi = mi_full[i]
        oof_mi = mi_oof[i]

        # Бесполезный признак
        if oof_mi < 0.001:
            logger.warning(
                "[oof_check] '%s' ОТБРОШЕН: OOF MI=%.5f < 0.001 (бесполезен)",
                col, oof_mi,
            )
            continue

        # Leakage check: MI падает на OOF более чем на 80%
        if full_mi > 0.01 and oof_mi < full_mi * 0.2:
            logger.warning(
                "[oof_check] '%s' ОТБРОШЕН: подозрение на leakage. "
                "full_MI=%.5f, OOF_MI=%.5f (падение %.0f%%)",
                col, full_mi, oof_mi, (1 - oof_mi / full_mi) * 100,
            )
            continue

        logger.info(
            "[oof_check] '%s' ОК: full_MI=%.5f, OOF_MI=%.5f",
            col, full_mi, oof_mi,
        )
        good_cols.append(col)

    if not good_cols:
        logger.warning("[oof_check] Все признаки отброшены! Возвращаем исходный набор.")
        return feature_cols

    if len(good_cols) < len(feature_cols):
        logger.info(
            "[oof_check] Отфильтровано: %d → %d признаков",
            len(feature_cols), len(good_cols),
        )

    return good_cols


# ---------------------------------------------------------------------------
# Многоуровневый fallback
# ---------------------------------------------------------------------------
#
# Идея: если LLM-пайплайн сломался, не обязательно отдавать случайные числа.
# Пробуем всё более простые стратегии, первая прошедшая структурную проверку
# становится финальным сабмитом.
#
# Уровни (от "осмысленных" к "гарантированным"):
#   L1 — frequency encoding до 5 категориальных колонок из доп.таблиц
#   L2 — прямой маппинг до 5 числовых колонок из доп.таблиц по id
#   L3 — случайные нормальные числа (работает всегда)
#
# Каждая стратегия:
#   - возвращает (train_df, test_df, feat_names) или None если неприменима
#   - НЕ пишет файлы сама
# Функция save_fallback() перебирает уровни, пишет первый валидный.

def _find_id_col(src_train: pd.DataFrame, src_test: pd.DataFrame) -> str:
    """Берёт первую общую колонку train/test — скорее всего это id."""
    for c in src_train.columns:
        if c in src_test.columns:
            return c
    return src_train.columns[0]


def _aux_tables(data_dir: Path) -> list[tuple[str, pd.DataFrame]]:
    """Читает все csv из data/ кроме train/test/hidden и возвращает (имя, df)."""
    out: list[tuple[str, pd.DataFrame]] = []
    if not data_dir.exists():
        return out
    for p in sorted(data_dir.iterdir()):
        if p.suffix.lower() != ".csv":
            continue
        if p.name in ("train.csv", "test.csv"):
            continue
        try:
            out.append((p.name, pd.read_csv(p)))
        except Exception as e:
            logger.warning("[fallback] Не удалось прочитать %s: %s", p.name, e)
    return out


def _fallback_l1_freq_encoding(
    src_train: pd.DataFrame,
    src_test: pd.DataFrame,
    id_col: str,
) -> tuple[pd.DataFrame, pd.DataFrame, list[str]] | None:
    """L1: частотное кодирование до 5 категориальных колонок из доп.таблиц."""
    aux = _aux_tables(DATA_DIR)
    if not aux:
        return None

    candidates: list[tuple[str, str, pd.DataFrame]] = []  # (aux_name, col_name, df)
    for aname, adf in aux:
        if id_col not in adf.columns:
            continue
        for col in adf.columns:
            if col == id_col:
                continue
            # Категориальная = object/string с разумным количеством уникальных значений
            if adf[col].dtype == object or pd.api.types.is_string_dtype(adf[col]):
                nuniq = adf[col].nunique(dropna=True)
                if 2 <= nuniq <= 50:
                    candidates.append((aname, col, adf))

    if not candidates:
        return None

    candidates = candidates[:5]
    train_out = src_train.copy()
    test_out = src_test.copy()
    feat_names: list[str] = []

    for aname, col, adf in candidates:
        feat_name = f"fb_freq_{col}"[:50]
        if feat_name in feat_names:
            continue
        freq = adf[col].value_counts(normalize=True).to_dict()
        id_to_cat = adf.drop_duplicates(id_col).set_index(id_col)[col]
        train_cat = train_out[id_col].map(id_to_cat)
        test_cat = test_out[id_col].map(id_to_cat)
        train_out[feat_name] = train_cat.map(freq).fillna(0.0).astype(float)
        test_out[feat_name] = test_cat.map(freq).fillna(0.0).astype(float)
        feat_names.append(feat_name)
        if len(feat_names) >= 5:
            break

    if not feat_names:
        return None
    return train_out, test_out, feat_names


def _fallback_l2_numeric_map(
    src_train: pd.DataFrame,
    src_test: pd.DataFrame,
    id_col: str,
) -> tuple[pd.DataFrame, pd.DataFrame, list[str]] | None:
    """L2: прямой маппинг до 5 числовых колонок из доп.таблиц."""
    aux = _aux_tables(DATA_DIR)
    if not aux:
        return None

    candidates: list[tuple[str, str, pd.DataFrame]] = []
    for aname, adf in aux:
        if id_col not in adf.columns:
            continue
        for col in adf.columns:
            if col == id_col:
                continue
            if pd.api.types.is_numeric_dtype(adf[col]):
                # Исключаем константы
                if adf[col].nunique(dropna=True) >= 2:
                    candidates.append((aname, col, adf))

    if not candidates:
        return None

    candidates = candidates[:5]
    train_out = src_train.copy()
    test_out = src_test.copy()
    feat_names: list[str] = []

    for aname, col, adf in candidates:
        feat_name = f"fb_num_{col}"[:50]
        if feat_name in feat_names:
            continue
        id_to_val = adf.drop_duplicates(id_col).set_index(id_col)[col]
        med = float(pd.to_numeric(id_to_val, errors="coerce").median()) if len(id_to_val) else 0.0
        if not np.isfinite(med):
            med = 0.0
        train_out[feat_name] = pd.to_numeric(
            train_out[id_col].map(id_to_val), errors="coerce"
        ).fillna(med).astype(float)
        test_out[feat_name] = pd.to_numeric(
            test_out[id_col].map(id_to_val), errors="coerce"
        ).fillna(med).astype(float)
        feat_names.append(feat_name)
        if len(feat_names) >= 5:
            break

    if not feat_names:
        return None
    return train_out, test_out, feat_names


def _fallback_l3_random(
    src_train: pd.DataFrame,
    src_test: pd.DataFrame,
    id_col: str,  # noqa: ARG001 — сигнатура унифицирована со стратегиями
) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    """L3: случайные нормальные числа. Работает всегда."""
    rng = np.random.default_rng(42)
    train_out = src_train.copy()
    test_out = src_test.copy()
    feat_names = [f"fb_rand_{i+1}" for i in range(5)]
    for fname in feat_names:
        train_out[fname] = rng.normal(size=len(train_out))
        test_out[fname] = rng.normal(size=len(test_out))
    return train_out, test_out, feat_names


def _validate_fallback_output(
    src_train: pd.DataFrame,
    src_test: pd.DataFrame,
    train_out: pd.DataFrame,
    test_out: pd.DataFrame,
    feat_names: list[str],
) -> tuple[bool, str]:
    """Быстрая проверка инвариантов check_submission. True если OK."""
    if len(train_out) != len(src_train):
        return False, f"train rows: {len(train_out)} != {len(src_train)}"
    if len(test_out) != len(src_test):
        return False, f"test rows: {len(test_out)} != {len(src_test)}"
    if not (1 <= len(feat_names) <= 5):
        return False, f"feature count out of range: {len(feat_names)}"
    for c in src_train.columns:
        if c not in train_out.columns:
            return False, f"train missing original col: {c}"
    for c in src_test.columns:
        if c not in test_out.columns:
            return False, f"test missing original col: {c}"
    train_feats = [c for c in train_out.columns if c not in src_train.columns]
    test_feats = [c for c in test_out.columns if c not in src_test.columns]
    if train_feats != test_feats:
        return False, f"feature mismatch: train={train_feats} test={test_feats}"
    for c in feat_names:
        if train_out[c].isna().all() or test_out[c].isna().all():
            return False, f"feature all-NaN: {c}"
    if not train_out.columns.is_unique or not test_out.columns.is_unique:
        return False, "duplicate column names"
    return True, ""


def save_fallback() -> None:
    """Многоуровневый аварийный сабмит.

    Перебирает стратегии от "осмысленной" к "гарантированной". Первая,
    прошедшая валидацию, пишется в output/. Если всё сломалось — L3 (random)
    всегда проходит.
    """
    logger.warning("[orchestrator] Запуск многоуровневого fallback...")
    src_train = pd.read_csv(DATA_DIR / "train.csv")
    src_test = pd.read_csv(DATA_DIR / "test.csv")
    id_col = _find_id_col(src_train, src_test)
    logger.info("[fallback] id_col=%s, train=%s, test=%s", id_col, src_train.shape, src_test.shape)

    strategies = [
        ("L1_freq_encoding", _fallback_l1_freq_encoding),
        ("L2_numeric_map",   _fallback_l2_numeric_map),
        ("L3_random",        _fallback_l3_random),
    ]

    for level_name, strategy in strategies:
        try:
            result = strategy(src_train, src_test, id_col)
        except Exception as e:
            logger.warning("[fallback] %s упал: %s", level_name, e)
            continue
        if result is None:
            logger.info("[fallback] %s неприменим (нет подходящих колонок)", level_name)
            continue

        train_out, test_out, feat_names = result
        ok, reason = _validate_fallback_output(src_train, src_test, train_out, test_out, feat_names)
        if not ok:
            logger.warning("[fallback] %s не прошёл валидацию: %s", level_name, reason)
            continue

        OUTPUT_DIR.mkdir(exist_ok=True)
        train_out.to_csv(OUTPUT_DIR / "train.csv", index=False)
        test_out.to_csv(OUTPUT_DIR / "test.csv", index=False)
        logger.info("[fallback] %s ПРИНЯТ. Признаки: %s", level_name, feat_names)
        return

    # Сюда не должны попасть — L3 всегда работает. Но на всякий случай:
    logger.error("[fallback] Все уровни fallback провалились — критично!")


def _deterministic_compare(stats_r1: dict[str, Any], stats_r2: dict[str, Any]) -> int:
    """Детерминированное сравнение двух наборов признаков по метрикам compute_stats.

    Критерии (по приоритету):
      1. Дисквалификация за leakage: |pearson| > 0.9 или MI > 0.5
      2. Суммарный MI (без leakage-фич) — чем больше, тем лучше
      3. При равенстве MI — суммарный |pearson|

    Returns:
        1 или 2 — номер лучшего раунда.
    """
    def _score(stats: dict[str, Any]) -> tuple[bool, float, float]:
        features = stats.get("features", {})
        has_leakage = False
        total_mi = 0.0
        total_pearson = 0.0
        for _name, finfo in features.items():
            mi = abs(float(finfo.get("mutual_info", 0.0)))
            pearson = abs(float(finfo.get("pearson", 0.0)))
            if pearson > 0.9 or mi > 0.5:
                has_leakage = True
                continue  # не считаем leaky-фичи в сумму
            total_mi += mi
            total_pearson += pearson
        return has_leakage, total_mi, total_pearson

    leak1, mi1, p1 = _score(stats_r1)
    leak2, mi2, p2 = _score(stats_r2)

    logger.info(
        "[compare] R1: leakage=%s, sum_MI=%.4f, sum_|pearson|=%.4f", leak1, mi1, p1,
    )
    logger.info(
        "[compare] R2: leakage=%s, sum_MI=%.4f, sum_|pearson|=%.4f", leak2, mi2, p2,
    )

    # Leakage дисквалифицирует
    if leak1 and not leak2:
        logger.info("[compare] Раунд 1 дисквалифицирован за leakage → раунд 2")
        return 2
    if leak2 and not leak1:
        logger.info("[compare] Раунд 2 дисквалифицирован за leakage → раунд 1")
        return 1

    # Суммарный MI
    if abs(mi1 - mi2) > 1e-6:
        winner = 1 if mi1 > mi2 else 2
        logger.info("[compare] По суммарному MI → раунд %d", winner)
        return winner

    # При равенстве MI — по pearson
    winner = 1 if p1 >= p2 else 2
    logger.info("[compare] MI равны, по суммарному |pearson| → раунд %d", winner)
    return winner


def _log_step_budget(step: str, took: float) -> bool:
    """Логирует факт исполнения шага и сравнивает с бюджетом.

    Returns:
        True если уложились в бюджет, False если перебор.
    """
    budget = STEP_BUDGETS.get(step, 9999)
    if took > budget:
        overshoot = took - budget
        logger.warning(
            "[orchestrator] ⚠ шаг '%s' занял %.0f сек (бюджет %d сек, перебор +%.0f сек)",
            step, took, budget, overshoot,
        )
        return False
    logger.info("[orchestrator] ✓ шаг '%s' занял %.0f сек (бюджет %d сек)", step, took, budget)
    return True


def run_pipeline(llm: GigaChat) -> None:
    """Запускает полный пайплайн оркестратора."""
    t_start = time.monotonic()
    step_overruns = 0  # сколько шагов перебрали бюджет — влияет на решение о раунде 2

    def elapsed() -> float:
        return time.monotonic() - t_start

    def remaining() -> float:
        return TIME_BUDGET - elapsed()

    logger.info("=" * 60)
    logger.info("[orchestrator] Запуск пайплайна. Бюджет: %d сек.", TIME_BUDGET)
    logger.info("[orchestrator] Бюджеты шагов: %s", STEP_BUDGETS)
    logger.info("=" * 60)

    # ------------------------------------------------------------------
    # ШАГ 1: Аналитик
    # ------------------------------------------------------------------
    logger.info("[orchestrator] ШАГ 1: Запуск аналитика (осталось ~%.0f сек.)", remaining())
    t_step = time.monotonic()
    analyst_report = run_analyst(llm, task="исследуй данные с нуля")
    if not _log_step_budget("analyst", time.monotonic() - t_step):
        step_overruns += 1

    id_col = analyst_report.get("id_column", "client_id")
    target_col = analyst_report.get("target_column", "target")
    logger.info(
        "[orchestrator] Аналитик завершён за %.0f сек. id='%s', target='%s'",
        elapsed(), id_col, target_col,
    )

    # Состояние раундов
    round_data: dict[int, dict[str, Any]] = {}

    # ------------------------------------------------------------------
    # РАУНД 1
    # ------------------------------------------------------------------
    logger.info("[orchestrator] --- РАУНД 1 (осталось ~%.0f сек.) ---", remaining())

    # ШАГ 2: Генератор
    logger.info("[orchestrator] ШАГ 2: Генератор (раунд 1)")
    t_step = time.monotonic()
    features_r1 = run_generator(llm, analyst_report, round_num=1)
    if not _log_step_budget("generator", time.monotonic() - t_step):
        step_overruns += 1
    if not features_r1:
        logger.error("[orchestrator] Генератор не вернул признаки. Сохраняю fallback.")
        save_fallback()
        return
    logger.info("[orchestrator] Генератор предложил %d признаков за %.0f сек.", len(features_r1), elapsed())

    # ШАГ 3: Кодер — build_features
    logger.info("[orchestrator] ШАГ 3: Кодер — build_features (раунд 1)")
    t_step = time.monotonic()
    df_train_r1, df_test_r1, err = build_features(llm, analyst_report, features_r1)
    if not _log_step_budget("build_features", time.monotonic() - t_step):
        step_overruns += 1

    if df_train_r1 is None:
        logger.error("[orchestrator] Кодер не смог построить признаки раунда 1: %s", err)
        logger.warning("[orchestrator] Сохраняю fallback-признаки.")
        save_fallback()
        return

    feature_cols_r1 = _get_feature_cols(df_train_r1, id_col, target_col)
    logger.info("[orchestrator] Построено признаков (раунд 1): %d — %s", len(feature_cols_r1), feature_cols_r1)

    # OOF-фильтрация: убираем leaky/бесполезные признаки
    feature_cols_r1 = _oof_quality_check(df_train_r1, target_col, feature_cols_r1)
    logger.info("[orchestrator] После OOF-фильтрации (раунд 1): %d — %s", len(feature_cols_r1), feature_cols_r1)

    # ШАГ 4: Кодер — compute_stats (на отфильтрованных признаках)
    logger.info("[orchestrator] ШАГ 4: compute_stats (раунд 1)")
    # Оставляем только отфильтрованные колонки в df_train/df_test
    keep_cols_r1 = [id_col, target_col] + feature_cols_r1
    df_train_r1 = df_train_r1[[c for c in keep_cols_r1 if c in df_train_r1.columns]]
    df_test_r1 = df_test_r1[[c for c in [id_col] + feature_cols_r1 if c in df_test_r1.columns]]
    t_step = time.monotonic()
    stats_r1 = compute_stats(df_train_r1, target_col, id_col)
    _log_step_budget("compute_stats", time.monotonic() - t_step)

    # ШАГ 5: Критик
    logger.info("[orchestrator] ШАГ 5: Критик оценивает раунд 1 (осталось ~%.0f сек.)", remaining())
    t_step = time.monotonic()
    eval_r1 = run_critic(llm, stats_r1, round_num=1)
    if not _log_step_budget("critic", time.monotonic() - t_step):
        step_overruns += 1

    round_data[1] = {
        "df_train": df_train_r1,
        "df_test": df_test_r1,
        "feature_cols": feature_cols_r1,
        "stats": stats_r1,
        "eval": eval_r1,
    }

    overall_r1 = eval_r1.get("overall_score", 0.5)
    logger.info(
        "[orchestrator] Раунд 1 завершён за %.0f сек. overall_score=%.2f",
        elapsed(), overall_r1,
    )

    # ------------------------------------------------------------------
    # Проверка: запускать ли раунд 2?
    # ------------------------------------------------------------------
    time_left = remaining()
    need_r2 = eval_r1.get("need_second_round", False)
    logger.info(
        "[orchestrator] Остаток времени: %.0f сек. Нужен раунд 2: %s (overall=%.2f, step_overruns=%d)",
        time_left, need_r2, overall_r1, step_overruns,
    )

    # Если раунд 1 уже перебрал бюджеты нескольких шагов, раунд 2 рискованно —
    # в пайплайне может не остаться времени даже на save. Лучше закрепить r1.
    if step_overruns >= 2:
        logger.warning(
            "[orchestrator] Раунд 1 перебрал бюджеты %d шагов — пропускаем раунд 2 для надёжности",
            step_overruns,
        )
        need_r2 = False

    if need_r2 and time_left >= TIME_FOR_ROUND2:
        # ------------------------------------------------------------------
        # РАУНД 2
        # ------------------------------------------------------------------
        logger.info("[orchestrator] --- РАУНД 2 (осталось ~%.0f сек.) ---", time_left)

        # Уточнение у аналитика если критик попросил
        if eval_r1.get("need_clarification") and eval_r1.get("clarification_question"):
            q = eval_r1["clarification_question"]
            logger.info("[orchestrator] ШАГ 6: Уточняющий вопрос аналитику: %s", q[:80])
            clarification = run_analyst(llm, task=q)
            analyst_report["clarification"] = clarification
            logger.info("[orchestrator] Аналитик ответил на уточнение")

        # ШАГ 7: Генератор (с фидбеком)
        logger.info("[orchestrator] ШАГ 7: Генератор (раунд 2, с фидбеком критика)")
        critic_feedback = {
            "feedback": eval_r1.get("feedback_for_generator", ""),
            "weak_features": [
                name for name, s in eval_r1.get("scores", {}).items()
                if s.get("verdict") in ("weak", "suspicious_leakage")
            ],
        }
        t_step = time.monotonic()
        features_r2 = run_generator(llm, analyst_report, critic_feedback=critic_feedback, round_num=2)
        _log_step_budget("generator", time.monotonic() - t_step)

        if not features_r2:
            logger.warning("[orchestrator] Генератор раунда 2 не вернул признаков, пропускаем раунд 2")
        else:
            # ШАГ 8: Кодер
            logger.info("[orchestrator] ШАГ 8: Кодер — build_features (раунд 2)")
            t_step = time.monotonic()
            df_train_r2, df_test_r2, err2 = build_features(llm, analyst_report, features_r2)
            _log_step_budget("build_features", time.monotonic() - t_step)

            if df_train_r2 is None:
                logger.warning("[orchestrator] Кодер раунда 2 упал: %s. Используем раунд 1.", err2)
            else:
                feature_cols_r2 = _get_feature_cols(df_train_r2, id_col, target_col)
                logger.info("[orchestrator] Признаки раунда 2: %d — %s", len(feature_cols_r2), feature_cols_r2)

                # OOF-фильтрация раунда 2
                feature_cols_r2 = _oof_quality_check(df_train_r2, target_col, feature_cols_r2)
                logger.info("[orchestrator] После OOF-фильтрации (раунд 2): %d — %s", len(feature_cols_r2), feature_cols_r2)
                keep_cols_r2 = [id_col, target_col] + feature_cols_r2
                df_train_r2 = df_train_r2[[c for c in keep_cols_r2 if c in df_train_r2.columns]]
                df_test_r2 = df_test_r2[[c for c in [id_col] + feature_cols_r2 if c in df_test_r2.columns]]

                # ШАГ 9: compute_stats (на отфильтрованных признаках)
                logger.info("[orchestrator] ШАГ 9: compute_stats (раунд 2)")
                t_step = time.monotonic()
                stats_r2 = compute_stats(df_train_r2, target_col, id_col)
                _log_step_budget("compute_stats", time.monotonic() - t_step)

                # ШАГ 10: Критик оценивает раунд 2 и сравнивает
                logger.info("[orchestrator] ШАГ 10: Критик оценивает раунд 2 (осталось ~%.0f сек.)", remaining())
                t_step = time.monotonic()
                eval_r2 = run_critic(llm, stats_r2, round_num=2)
                _log_step_budget("critic", time.monotonic() - t_step)

                round_data[2] = {
                    "df_train": df_train_r2,
                    "df_test": df_test_r2,
                    "feature_cols": feature_cols_r2,
                    "stats": stats_r2,
                    "eval": eval_r2,
                }

                overall_r2 = eval_r2.get("overall_score", 0.5)
                logger.info(
                    "[orchestrator] Раунд 2 завершён за %.0f сек. overall_score=%.2f",
                    elapsed(), overall_r2,
                )

                # Сравниваем раунды детерминированно по метрикам (без LLM-вызова)
                logger.info("[orchestrator] Детерминированное сравнение раундов по метрикам...")
                winner = _deterministic_compare(stats_r1, stats_r2)
                logger.info("[orchestrator] Победитель: раунд %d", winner)

                # Перезаписываем раунд 1 если победил раунд 2
                if winner == 2:
                    round_data[1] = round_data[2]
                    logger.info("[orchestrator] Используем признаки раунда 2")
    else:
        if not need_r2:
            logger.info("[orchestrator] Раунд 2 не нужен (overall_score достаточно высокий)")
        else:
            logger.warning("[orchestrator] Недостаточно времени для раунда 2 (осталось %.0f сек.)", time_left)

    # ------------------------------------------------------------------
    # ШАГ 11: Save
    # ------------------------------------------------------------------
    best = round_data.get(1)
    if best is None:
        logger.error("[orchestrator] Нет данных для сохранения! Сохраняю fallback.")
        save_fallback()
        return

    logger.info("[orchestrator] ШАГ 11: Сохранение финальных признаков (осталось ~%.0f сек.)", remaining())
    t_step = time.monotonic()
    try:
        save(
            df_train=best["df_train"],
            df_test=best["df_test"],
            analyst_report=analyst_report,
            feature_cols=best["feature_cols"],
        )
        _log_step_budget("save", time.monotonic() - t_step)
    except Exception as e:
        logger.error("[orchestrator] Ошибка при сохранении: %s. Сохраняю fallback.", e)
        save_fallback()
        return

    logger.info("=" * 60)
    logger.info("[orchestrator] Пайплайн завершён за %.0f сек.", elapsed())
    logger.info("[orchestrator] Финальные признаки: %s", best["feature_cols"][:5])
    logger.info("=" * 60)
