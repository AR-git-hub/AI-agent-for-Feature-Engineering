"""Агент 4: сборка выходных таблиц для CatBoost и запись в output/.

Формат, который ждёт CatBoost (и check_submission.py):
  output/train.csv  — [id_col, target_col, feat_1, …, feat_k],  k ≤ 5
  output/test.csv   — [id_col, feat_1, …, feat_k],              k ≤ 5

Источники данных:
  ctx.train_frame / ctx.test_frame   — собранный датасет (агент 1)
  ctx.feature_matrix_train / _test   — сгенерированные фичи (агент 2)
  ctx.selected_feature_names         — отобранные имена (агент 3, ≤ 5 штук)
"""
from __future__ import annotations

import logging

import pandas as pd

from src.mas.context import RunContext
from src.mas.tools import output_tools

logger = logging.getLogger("mas.agent.answer")

MAX_FEATURES = 5


def build_train_output(ctx: RunContext) -> pd.DataFrame:
    """
    Собрать output/train.csv для CatBoost:
    все исходные колонки data/train.csv + сгенерированные фичи.

    check_submission.py требует присутствия ВСЕХ колонок из data/train.csv.
    Фичи — это всё что не входит в исходный набор колонок.
    """
    if ctx.train_frame is None or ctx.feature_matrix_train is None:
        logger.error("build_train_output: train_frame или feature_matrix_train отсутствует")
        return pd.DataFrame()

    feat_cols = [
        c for c in ctx.selected_feature_names
        if c in ctx.feature_matrix_train.columns
    ][:MAX_FEATURES]

    # Берём все исходные колонки из data/train.csv, которые есть в train_frame
    reserved = ctx.input_train_cols or (
        [c for c in (ctx.id_col, ctx.target_col) if c]
    )
    base_cols = [c for c in reserved if c in ctx.train_frame.columns]

    parts: list[pd.DataFrame] = []
    if base_cols:
        parts.append(ctx.train_frame[base_cols].reset_index(drop=True))
    if feat_cols:
        parts.append(ctx.feature_matrix_train[feat_cols].reset_index(drop=True))

    if not parts:
        return pd.DataFrame()

    return pd.concat(parts, axis=1)


def build_test_output(ctx: RunContext) -> pd.DataFrame:
    """
    Собрать output/test.csv для CatBoost:
    все исходные колонки data/test.csv + сгенерированные фичи (без таргета).

    check_submission.py требует присутствия ВСЕХ колонок из data/test.csv.
    """
    if ctx.test_frame is None or ctx.feature_matrix_test is None:
        logger.error("build_test_output: test_frame или feature_matrix_test отсутствует")
        return pd.DataFrame()

    feat_cols = [
        c for c in ctx.selected_feature_names
        if c in ctx.feature_matrix_test.columns
    ][:MAX_FEATURES]

    # Берём все исходные колонки из data/test.csv, которые есть в test_frame
    reserved = ctx.input_test_cols or (
        [ctx.id_col] if ctx.id_col else []
    )
    base_cols = [c for c in reserved if c in ctx.test_frame.columns]

    parts: list[pd.DataFrame] = []
    if base_cols:
        parts.append(ctx.test_frame[base_cols].reset_index(drop=True))
    if feat_cols:
        parts.append(ctx.feature_matrix_test[feat_cols].reset_index(drop=True))

    if not parts:
        return pd.DataFrame()

    return pd.concat(parts, axis=1)


def validate_outputs(train: pd.DataFrame, test: pd.DataFrame) -> None:
    """
    Минимальные проверки перед записью:
    - одинаковые фичи в train и test;
    - не более MAX_FEATURES числовых фичей;
    - нет NaN (дефолтный CatBoost падает на NaN).
    Организаторские проверки — в check_submission.py.
    """
    if train.empty or test.empty:
        logger.warning("validate_outputs: один из DataFrames пуст, пропускаем проверку.")
        return

    # target присутствует только в train — это ожидаемо, исключаем из проверки.
    feat_train = set(train.columns)
    feat_test = set(test.columns)
    unexpected_in_test = feat_test - feat_train
    missing_in_test = feat_train - feat_test

    # Убираем target из ожидаемых расхождений (он всегда только в train).
    target_candidates = {"target", "y", "label"}
    expected_diff = {c for c in missing_in_test if c.lower() in target_candidates}
    real_missing = missing_in_test - expected_diff

    if unexpected_in_test or real_missing:
        logger.warning(
            "Колонки train/test расходятся: лишние в test=%s, отсутствующие в test=%s",
            unexpected_in_test,
            real_missing,
        )

    nan_train = train.isnull().sum().sum()
    nan_test = test.isnull().sum().sum()
    if nan_train:
        logger.warning("validate_outputs: %d NaN в train.csv", nan_train)
    if nan_test:
        logger.warning("validate_outputs: %d NaN в test.csv", nan_test)


class AnswerAgent:
    """Агент 4: собирает входные данные для CatBoost и сохраняет в output/."""

    def run(self, ctx: RunContext) -> RunContext:
        train = build_train_output(ctx)
        test = build_test_output(ctx)
        validate_outputs(train, test)

        ctx.train_features = train
        ctx.test_features = test

        if not train.empty and not test.empty:
            reserved = set(ctx.input_train_cols) | set(ctx.input_test_cols)
            output_tools.save_submission(
                ctx.output_dir, train, test,
                id_col=ctx.id_col,
                target_col=ctx.target_col,
                reserved_cols=reserved or None,
            )
            logger.info(
                "Сохранено: output/train.csv %s, output/test.csv %s",
                train.shape, test.shape,
            )
        else:
            output_tools.ensure_output_dir(ctx.output_dir)
            logger.error("Не удалось собрать выходные таблицы — output/ пуст.")

        return ctx
