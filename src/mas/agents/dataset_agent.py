"""Агент 1: сборка датасета.

Поток:
1. Загружаем все CSV из data/.
2. Базовый EDA каждой таблицы (для внутренней диагностики).
3. GigaChat читает readme + схемы таблиц → возвращает JSON-план мержей.
4. Выполняем request_merge для каждой связи (с диагностикой → ctx.merge_reports).
   Fallback: авто-определение ключей, если GigaChat не вернул парсируемый план.
5. Приведение типов / нормализация категорий.
6. Строим компактный EDA-отчёт по train_frame → ctx.eda_report (для агента 2).

ctx.configs_dir зарезервирован под pytest; агент его не читает.
"""
from __future__ import annotations

import json
import logging
import re

import pandas as pd
from gigachat.models import Chat, Messages, MessagesRole

from src.mas.context import RunContext
from src.mas.llm import chat_logged, gigachat_client
from src.mas.prompts import MERGE_PLANNING_SYSTEM, build_merge_planning_prompt
from src.mas.tools import data_tools, eda_tools

logger = logging.getLogger("mas.agent.dataset")

_ID_KEYWORDS = {"id", "index", "object_id", "client_id", "user_id", "customerid", "rowid"}
_JSON_RE = re.compile(r"```json\s*(.*?)```", re.DOTALL | re.IGNORECASE)


class DatasetAgent:
    def run(self, ctx: RunContext) -> RunContext:
        self._load_raw_tables(ctx)
        self._run_eda_on_loaded_tables(ctx)
        self._assemble_dataset(ctx)
        self._process_dataset(ctx)
        self._build_eda_report_for_llm(ctx)
        return ctx

    # ------------------------------------------------------------------
    # Шаг 1 — загрузка
    # ------------------------------------------------------------------

    def _load_raw_tables(self, ctx: RunContext) -> None:
        ctx.readme_text = data_tools.read_readme(ctx.data_dir)
        for name in data_tools.list_csv_tables(ctx.data_dir):
            try:
                df = data_tools.load_csv(ctx.data_dir, name)
            except Exception:
                df = data_tools.load_csv(ctx.data_dir, name, sep=None, engine="python")
            ctx.tables[name] = self._drop_unnamed_index_cols(df)
        logger.info("Загружено таблиц: %d", len(ctx.tables))

        # Запоминаем исходные колонки — check_submission требует их в output/
        if "train.csv" in ctx.tables:
            ctx.input_train_cols = list(ctx.tables["train.csv"].columns)
        if "test.csv" in ctx.tables:
            ctx.input_test_cols = list(ctx.tables["test.csv"].columns)

    @staticmethod
    def _drop_unnamed_index_cols(df: pd.DataFrame) -> pd.DataFrame:
        """Удаляет колонки вида 'Unnamed: N' — артефакт сохранения CSV с индексом."""
        unnamed = [c for c in df.columns if str(c).startswith("Unnamed:")]
        if unnamed:
            logger.info("Удалены Unnamed-колонки: %s", unnamed)
            df = df.drop(columns=unnamed)
        return df

    def _run_eda_on_loaded_tables(self, ctx: RunContext) -> None:
        ctx.eda_basic_by_table.clear()
        for name, df in ctx.tables.items():
            try:
                ctx.eda_basic_by_table[name] = eda_tools.basic_eda(df)
            except Exception:
                ctx.eda_basic_by_table[name] = pd.DataFrame()

    # ------------------------------------------------------------------
    # Шаг 2 — сборка: GigaChat планирует мержи, request_merge выполняет
    # ------------------------------------------------------------------

    def _assemble_dataset(self, ctx: RunContext) -> None:
        train_raw = ctx.tables.get("train.csv")
        test_raw = ctx.tables.get("test.csv")
        if train_raw is None or test_raw is None:
            logger.error("В data/ не найден train.csv или test.csv")
            ctx.schema_notes = "Ошибка: отсутствует train.csv или test.csv"
            return

        ctx.target_col = self._detect_target_col(train_raw, test_raw)
        ctx.id_col = self._detect_id_col(train_raw, test_raw)
        logger.info("target_col=%r  id_col=%r", ctx.target_col, ctx.id_col)

        data_tools.init_registry(ctx.tables)
        ctx.merge_reports.clear()

        extras = [n for n in ctx.tables if n not in ("train.csv", "test.csv")]
        if not extras:
            ctx.train_frame = train_raw.reset_index(drop=True)
            ctx.test_frame = test_raw.reset_index(drop=True)
            ctx.schema_notes = f"train {train_raw.shape}, test {test_raw.shape} — доп. таблиц нет."
            logger.info(ctx.schema_notes)
            return

        # ── GigaChat планирует мержи ───────────────────────────────────
        merge_plan = self._plan_merges_with_llm(ctx, extras)

        # ── Если GigaChat не дал план — авто-fallback ──────────────────
        if not merge_plan:
            logger.warning("GigaChat не вернул план мержей — используем авто-определение ключей.")
            merge_plan = self._fallback_merge_plan(ctx, extras)

        train_name, test_name = "train.csv", "test.csv"
        for spec in merge_plan:
            rt = spec.get("right_table", "")
            lk = spec.get("left_key", "")
            rk = spec.get("right_key", lk)
            how = spec.get("how", "left")

            if not rt or not lk:
                logger.warning("Пропускаем неполный мерж-spec: %s", spec)
                continue

            result_train = f"train_merged_{rt}"
            result_test  = f"test_merged_{rt}"

            rep_tr = data_tools.request_merge(train_name, rt, lk, rk, how, result_train)
            rep_te = data_tools.request_merge(test_name,  rt, lk, rk, how, result_test)
            ctx.merge_reports.extend([rep_tr, rep_te])

            # Обновляем цепочку только если оба мержа зарегистрировали результат
            tr_ok = "error" not in json.loads(rep_tr)
            te_ok = "error" not in json.loads(rep_te)
            if tr_ok and te_ok:
                train_name, test_name = result_train, result_test
            else:
                logger.warning(
                    "Мерж с '%s' не удался (train_ok=%s, test_ok=%s) — "
                    "продолжаем со старым train_name='%s'.",
                    rt, tr_ok, te_ok, train_name,
                )

        train = data_tools._load_table(train_name)
        test = data_tools._load_table(test_name)
        ctx.train_frame = train.reset_index(drop=True)
        ctx.test_frame = test.reset_index(drop=True)
        ctx.schema_notes = (
            f"train {train.shape}, test {test.shape}, "
            f"target={ctx.target_col!r}, id={ctx.id_col!r}, "
            f"мержей={len(ctx.merge_reports)}"
        )
        logger.info(ctx.schema_notes)

    # ── GigaChat: планирование мержей ─────────────────────────────────

    def _plan_merges_with_llm(self, ctx: RunContext, extras: list[str]) -> list[dict]:
        tables_info = "\n".join(
            f"{name}: {list(df.columns)}" for name, df in ctx.tables.items()
        )
        user_text = build_merge_planning_prompt(
            readme=ctx.readme_text,
            tables_info=tables_info,
        )
        payload = Chat(
            messages=[
                Messages(role=MessagesRole.SYSTEM, content=MERGE_PLANNING_SYSTEM),
                Messages(role=MessagesRole.USER, content=user_text),
            ]
        )
        try:
            with gigachat_client() as client:
                completion = chat_logged(client, payload, label="Agent1-MergePlan")
            content = completion.choices[0].message.content or ""
            return self._parse_merge_plan(content)
        except Exception as exc:
            logger.warning("GigaChat упал при планировании мержей: %s", exc)
            return []

    @staticmethod
    def _parse_merge_plan(content: str) -> list[dict]:
        """Извлекает JSON-массив из ответа GigaChat; возвращает [] если не удалось."""
        match = _JSON_RE.search(content)
        raw = match.group(1) if match else content.strip()
        try:
            plan = json.loads(raw)
            if isinstance(plan, list):
                logger.info("GigaChat вернул план мержей: %d операций", len(plan))
                return plan
        except json.JSONDecodeError:
            pass
        logger.warning("Не удалось распарсить JSON из ответа GigaChat:\n%s", content[:300])
        return []

    # ── Fallback: авто-определение ключей ─────────────────────────────

    def _fallback_merge_plan(self, ctx: RunContext, extras: list[str]) -> list[dict]:
        plan = []
        for name in extras:
            lk, rk = self._detect_merge_keys(
                data_tools._load_table("train.csv"),
                data_tools._load_table(name),
            )
            if lk:
                plan.append({"right_table": name, "left_key": lk, "right_key": rk, "how": "left"})
            else:
                logger.warning("Fallback: не найден ключ для %s — пропускаем.", name)
        return plan

    @staticmethod
    def _detect_merge_keys(left: pd.DataFrame, right: pd.DataFrame) -> tuple[str | None, str | None]:
        common = [c for c in left.columns if c in right.columns]
        if common:
            key = min(common, key=lambda c: right[c].duplicated().sum())
            return key, key
        for lc in left.columns:
            lvals = set(left[lc].dropna().astype(str).unique())
            for rc in right.columns:
                rvals = set(right[rc].dropna().astype(str).unique())
                if len(lvals & rvals) / max(len(lvals), 1) >= 0.5:
                    return lc, rc
        return None, None

    # ── Вспомогательные детекторы ─────────────────────────────────────

    @staticmethod
    def _detect_target_col(train: pd.DataFrame, test: pd.DataFrame) -> str | None:
        candidates = [c for c in train.columns if c not in test.columns]
        for c in candidates:
            if train[c].dropna().nunique() == 2:
                return c
        return candidates[0] if candidates else None

    @staticmethod
    def _detect_id_col(train: pd.DataFrame, test: pd.DataFrame) -> str | None:
        for col in test.columns:
            norm = col.lower().replace("_", "").replace("-", "")
            if norm in _ID_KEYWORDS or col.lower().startswith("id"):
                return col
        nuniq = test.nunique()
        return str(nuniq.idxmax()) if not nuniq.empty else None

    # ------------------------------------------------------------------
    # Шаг 3 — обработка типов, выбросы, пропуски
    # ------------------------------------------------------------------

    def _process_dataset(self, ctx: RunContext) -> None:
        if ctx.train_frame is None or ctx.test_frame is None:
            return
        ctx.train_frame = eda_tools._fix_dtypes(ctx.train_frame)
        ctx.test_frame = eda_tools._fix_dtypes(ctx.test_frame)
        ctx.train_frame, ctx.test_frame = self._winsorize_and_impute(
            ctx.train_frame, ctx.test_frame,
            exclude={ctx.target_col, ctx.id_col},
        )
        logger.info("Типы приведены, выбросы винзоризованы (99%%), пропуски заполнены.")

    @staticmethod
    def _winsorize_and_impute(
        train: pd.DataFrame,
        test: pd.DataFrame,
        exclude: set,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Винзоризация 99% + заполнение пропусков; границы считаются по train."""
        train = train.copy()
        test = test.copy()

        num_cols = [c for c in train.select_dtypes(include="number").columns if c not in exclude]
        cat_cols = [c for c in train.select_dtypes(exclude="number").columns if c not in exclude]

        for col in num_cols:
            lo = train[col].quantile(0.01)
            hi = train[col].quantile(0.99)
            train[col] = train[col].clip(lo, hi)
            if col in test.columns:
                test[col] = test[col].clip(lo, hi)
            median = train[col].median()
            train[col] = train[col].fillna(median)
            if col in test.columns:
                test[col] = test[col].fillna(median)

        for col in cat_cols:
            mode_vals = train[col].mode()
            fill_val = mode_vals.iloc[0] if not mode_vals.empty else "unknown"
            train[col] = train[col].fillna(fill_val)
            if col in test.columns:
                test[col] = test[col].fillna(fill_val)

        return train, test

    # ------------------------------------------------------------------
    # Шаг 4 — EDA-отчёт для агента 2
    # ------------------------------------------------------------------

    def _build_eda_report_for_llm(self, ctx: RunContext) -> None:
        """
        Компактный текстовый EDA по каждому признаку train_frame.
        Каждый признак получает:
          - статистическую строку (null%, mean/std/skew/corr или unique/top)
          - короткий аналитический текст: как признак связан с таргетом и что это значит
        """
        if ctx.train_frame is None:
            ctx.eda_report = "(train_frame недоступен)"
            return

        df, target = ctx.train_frame, ctx.target_col
        lines: list[str] = [f"=== EDA Report: train {df.shape} ==="]

        if target and target in df.columns:
            vc = df[target].value_counts(normalize=True)
            lines.append("Target: '%s' — %s" % (target, "  ".join(f"{v}={r:.1%}" for v, r in vc.items())))
        lines.append("")

        exclude = {target, ctx.id_col}
        num_cols = [c for c in df.select_dtypes(include="number").columns if c not in exclude]
        cat_cols = [c for c in df.select_dtypes(exclude="number").columns if c not in exclude]

        if num_cols:
            lines.append("NUMERIC FEATURES:")
        for col in num_cols:
            s = df[col]
            valid = s.dropna()
            corr_val: float | None = None
            corr_str = ""
            if target and target in df.columns and len(valid) > 1:
                try:
                    corr_val = float(valid.corr(df.loc[valid.index, target]))
                    corr_str = f" corr_target={corr_val:+.3f}"
                except Exception:
                    pass
            lines.append(
                f"  {col} [numeric] null={s.isna().mean():.1%}"
                f" mean={valid.mean():.4g} std={valid.std():.4g}"
                f" min={valid.min():.4g} max={valid.max():.4g}"
                f" skew={valid.skew():.2f}{corr_str}"
            )
            analysis = self._analyze_feature_text(col, s, corr_val, is_numeric=True)
            lines.append(f"    → {analysis}")

        if cat_cols:
            lines.append("\nCATEGORICAL FEATURES:")
        for col in cat_cols:
            s = df[col]
            vc = s.astype(str).value_counts(normalize=True).head(5)
            top = "  ".join(f"{v}({r:.0%})" for v, r in vc.items())
            lines.append(f"  {col} [categ] null={s.isna().mean():.1%} unique={s.nunique()} top: {top}")
            # Для категориальных считаем target-rate по группам
            corr_val = None
            analysis = self._analyze_feature_text(col, s, corr_val, is_numeric=False,
                                                   df=df, target=target)
            lines.append(f"    → {analysis}")

        ctx.eda_report = "\n".join(lines)
        logger.info("EDA-отчёт построен (%d строк).", len(lines))

    @staticmethod
    def _analyze_feature_text(
        col: str,
        s: pd.Series,
        corr: float | None,
        is_numeric: bool,
        df: pd.DataFrame | None = None,
        target: str | None = None,
    ) -> str:
        """Генерирует краткий текстовый анализ признака и его связи с таргетом."""
        parts: list[str] = []

        if is_numeric:
            if corr is not None:
                abs_corr = abs(corr)
                direction = "положительная" if corr > 0 else "отрицательная"
                if abs_corr >= 0.15:
                    parts.append(
                        f"Сильная {direction} линейная корреляция с таргетом ({corr:+.3f}) — "
                        f"признак, вероятно, несёт предиктивную силу"
                    )
                elif abs_corr >= 0.05:
                    parts.append(
                        f"Умеренная {direction} корреляция с таргетом ({corr:+.3f}) — "
                        f"признак может быть полезен в совокупности с другими"
                    )
                else:
                    parts.append(
                        f"Слабая линейная связь с таргетом ({corr:+.3f}) — "
                        f"нелинейные преобразования или взаимодействия могут вскрыть зависимость"
                    )

            skew = s.skew()
            if abs(skew) > 3:
                parts.append(
                    f"Сильная скошенность (skew={skew:.2f}) — log1p/sqrt может улучшить сигнал для CatBoost"
                )
            elif abs(skew) > 1:
                parts.append(f"Умеренная скошенность (skew={skew:.2f})")

            null_pct = s.isna().mean()
            if null_pct > 0.2:
                parts.append(f"Много пропусков ({null_pct:.0%}) — заполнено медианой; возможно, добавить флаг was_null")
            elif null_pct > 0.05:
                parts.append(f"Пропуски {null_pct:.0%} — заполнено медианой")

        else:
            unique = s.nunique()
            null_pct = s.isna().mean()

            if unique <= 2:
                parts.append("Бинарный признак — напрямую понятен CatBoost")
            elif unique <= 6:
                parts.append(f"Мало категорий ({unique}) — CatBoost хорошо справится без дополнительного кодирования")
            elif unique > 50:
                parts.append(
                    f"Высокая кардинальность ({unique} уник. значений) — "
                    f"CatBoost поддерживает OHE/TE, но стоит рассмотреть группировку редких значений"
                )
            else:
                parts.append(f"Категориальный, {unique} значений")

            if null_pct > 0.05:
                parts.append(f"Пропуски {null_pct:.0%} — заполнено модой")

            # Если есть данные о target, считаем разброс target rate по группам
            if df is not None and target and target in df.columns:
                try:
                    grp = df.groupby(col)[target].mean()
                    spread = float(grp.max() - grp.min())
                    if spread > 0.15:
                        parts.append(
                            f"Большой разброс target rate между категориями ({spread:.2f}) — "
                            f"признак имеет высокую дискриминирующую силу"
                        )
                    elif spread > 0.05:
                        parts.append(f"Умеренный разброс target rate ({spread:.2f}) — полезен для разбиения")
                    else:
                        parts.append(f"Малый разброс target rate ({spread:.2f}) — слабый разделитель сам по себе")
                except Exception:
                    pass

        return "; ".join(parts) if parts else "Нейтральный признак, дополнительный анализ не выявил явной связи с таргетом"
