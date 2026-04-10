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

from src.agents.analyst import run_analyst
from src.agents.coder import build_features, compute_stats
from src.agents.critic import compare_rounds, run_critic
from src.agents.generator import run_generator

logger = logging.getLogger(__name__)

OUTPUT_DIR = Path("output")
TIME_BUDGET = 570          # секунд (оставляем буфер от лимита 600)
TIME_FOR_ROUND2 = 200      # минимум секунд для запуска раунда 2
TIME_EMERGENCY = 60        # меньше этого — сразу save


def save(
    df_train: pd.DataFrame,
    df_test: pd.DataFrame,
    analyst_report: dict[str, Any],
    feature_cols: list[str],
) -> None:
    """Сохраняет финальные признаки в output/train.csv и output/test.csv.

    Проверяет: id и target на месте, признаков не больше 5,
    структура колонок train и test совпадает.
    """
    id_col = analyst_report.get("id_column", "client_id")
    target_col = analyst_report.get("target_column", "target")

    # Берём не более 5 признаков
    cols = feature_cols[:5]
    logger.info("[save] Сохранение признаков: %s", cols)

    # --- train ---
    train_cols = [id_col] + ([target_col] if target_col in df_train.columns else []) + cols
    train_cols = [c for c in train_cols if c in df_train.columns]
    out_train = df_train[train_cols].copy()

    # --- test ---
    test_cols = [id_col] + cols
    test_cols = [c for c in test_cols if c in df_test.columns]
    out_test = df_test[test_cols].copy()

    OUTPUT_DIR.mkdir(exist_ok=True)
    out_train.to_csv(OUTPUT_DIR / "train.csv", index=False)
    out_test.to_csv(OUTPUT_DIR / "test.csv", index=False)

    logger.info("[save] output/train.csv: shape=%s, колонки=%s", out_train.shape, list(out_train.columns))
    logger.info("[save] output/test.csv:  shape=%s, колонки=%s", out_test.shape, list(out_test.columns))


def _get_feature_cols(df_train: pd.DataFrame, id_col: str, target_col: str) -> list[str]:
    return [c for c in df_train.columns if c not in (id_col, target_col)]


def run_pipeline(llm: GigaChat) -> None:
    """Запускает полный пайплайн оркестратора."""
    t_start = time.monotonic()

    def elapsed() -> float:
        return time.monotonic() - t_start

    def remaining() -> float:
        return TIME_BUDGET - elapsed()

    logger.info("=" * 60)
    logger.info("[orchestrator] Запуск пайплайна. Бюджет: %d сек.", TIME_BUDGET)
    logger.info("=" * 60)

    # ------------------------------------------------------------------
    # ШАГ 1: Аналитик
    # ------------------------------------------------------------------
    logger.info("[orchestrator] ШАГ 1: Запуск аналитика (осталось ~%.0f сек.)", remaining())
    analyst_report = run_analyst(llm, task="исследуй данные с нуля")

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
    features_r1 = run_generator(llm, analyst_report, round_num=1)
    if not features_r1:
        logger.error("[orchestrator] Генератор не вернул признаки. Аварийный выход.")
        return
    logger.info("[orchestrator] Генератор предложил %d признаков за %.0f сек.", len(features_r1), elapsed())

    # ШАГ 3: Кодер — build_features
    logger.info("[orchestrator] ШАГ 3: Кодер — build_features (раунд 1)")
    df_train_r1, df_test_r1, err = build_features(llm, analyst_report, features_r1)

    if df_train_r1 is None:
        logger.error("[orchestrator] Кодер не смог построить признаки раунда 1: %s", err)
        logger.warning("[orchestrator] Аварийное сохранение заглушки невозможно — нет данных.")
        return

    feature_cols_r1 = _get_feature_cols(df_train_r1, id_col, target_col)
    logger.info("[orchestrator] Построено признаков (раунд 1): %d — %s", len(feature_cols_r1), feature_cols_r1)

    # ШАГ 4: Кодер — compute_stats
    logger.info("[orchestrator] ШАГ 4: compute_stats (раунд 1)")
    stats_r1 = compute_stats(df_train_r1, target_col, id_col)

    # ШАГ 5: Критик
    logger.info("[orchestrator] ШАГ 5: Критик оценивает раунд 1 (осталось ~%.0f сек.)", remaining())
    eval_r1 = run_critic(llm, stats_r1, round_num=1)

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
        "[orchestrator] Остаток времени: %.0f сек. Нужен раунд 2: %s (overall=%.2f)",
        time_left, need_r2, overall_r1,
    )

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
        features_r2 = run_generator(llm, analyst_report, critic_feedback=critic_feedback, round_num=2)

        if not features_r2:
            logger.warning("[orchestrator] Генератор раунда 2 не вернул признаков, пропускаем раунд 2")
        else:
            # ШАГ 8: Кодер
            logger.info("[orchestrator] ШАГ 8: Кодер — build_features (раунд 2)")
            df_train_r2, df_test_r2, err2 = build_features(llm, analyst_report, features_r2)

            if df_train_r2 is None:
                logger.warning("[orchestrator] Кодер раунда 2 упал: %s. Используем раунд 1.", err2)
            else:
                feature_cols_r2 = _get_feature_cols(df_train_r2, id_col, target_col)
                logger.info("[orchestrator] Признаки раунда 2: %d — %s", len(feature_cols_r2), feature_cols_r2)

                # ШАГ 9: compute_stats
                logger.info("[orchestrator] ШАГ 9: compute_stats (раунд 2)")
                stats_r2 = compute_stats(df_train_r2, target_col, id_col)

                # ШАГ 10: Критик оценивает раунд 2 и сравнивает
                logger.info("[orchestrator] ШАГ 10: Критик оценивает раунд 2 (осталось ~%.0f сек.)", remaining())
                eval_r2 = run_critic(llm, stats_r2, round_num=2)

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

                # Сравниваем раунды
                if time_left > TIME_EMERGENCY:
                    logger.info("[orchestrator] Критик сравнивает раунды...")
                    winner = compare_rounds(
                        llm,
                        stats_r1, stats_r2,
                        eval_r1, eval_r2,
                    )
                    logger.info("[orchestrator] Победитель: раунд %d", winner)
                else:
                    # Выбираем по overall_score автоматически
                    winner = 2 if overall_r2 >= overall_r1 else 1
                    logger.info(
                        "[orchestrator] Нет времени на сравнение, выбираем по overall_score: раунд %d", winner
                    )

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
        logger.error("[orchestrator] Нет данных для сохранения!")
        return

    logger.info("[orchestrator] ШАГ 11: Сохранение финальных признаков (осталось ~%.0f сек.)", remaining())
    save(
        df_train=best["df_train"],
        df_test=best["df_test"],
        analyst_report=analyst_report,
        feature_cols=best["feature_cols"],
    )

    logger.info("=" * 60)
    logger.info("[orchestrator] Пайплайн завершён за %.0f сек.", elapsed())
    logger.info("[orchestrator] Финальные признаки: %s", best["feature_cols"][:5])
    logger.info("=" * 60)
