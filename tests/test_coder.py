"""Тесты для src/agents/coder.py — проверяем exec_globals и compute_stats без LLM."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.agents.coder import _extract_code, _load_csv, _get_table_columns, compute_stats


# ---------------------------------------------------------------------------
# Фикстуры
# ---------------------------------------------------------------------------

@pytest.fixture
def simple_train():
    """Маленький train DataFrame с двумя признаками и таргетом."""
    rng = np.random.default_rng(42)
    n = 200
    return pd.DataFrame({
        "row_id": range(n),
        "feat_a": rng.normal(0, 1, n),
        "feat_b": rng.integers(0, 5, n).astype(float),
        "target": rng.integers(0, 2, n),
    })


# ---------------------------------------------------------------------------
# _extract_code
# ---------------------------------------------------------------------------

class TestExtractCode:
    def test_plain_code(self):
        raw = "x = 1\ny = 2"
        assert _extract_code(raw) == "x = 1\ny = 2"

    def test_python_fence(self):
        raw = "```python\nx = 1\n```"
        assert _extract_code(raw) == "x = 1"

    def test_generic_fence(self):
        raw = "```\nx = 1\n```"
        assert _extract_code(raw) == "x = 1"

    def test_strips_whitespace(self):
        raw = "  \n```python\n  x = 1  \n```\n  "
        assert _extract_code(raw) == "x = 1"


# ---------------------------------------------------------------------------
# exec_globals — проверяем что переменные доступны в exec-коде
# ---------------------------------------------------------------------------

class TestExecGlobals:
    """Проверяет, что все переменные из exec_globals доступны в exec."""

    def _make_globals(self, train: pd.DataFrame, test: pd.DataFrame) -> dict:
        return {
            "id_col": "row_id",
            "target_col": "target",
            "data_dir": "data/",
            "output_dir": "output/",
            "train": train.copy(),
            "test": test.copy(),
            "analysis_report": {},
            "separator": ",",
            "pd": pd,
            "np": np,
            "__builtins__": __builtins__,
        }

    def test_variables_accessible(self, simple_train):
        """Код может обращаться к train, test, id_col, target_col."""
        test_df = simple_train.drop(columns=["target"])
        g = self._make_globals(simple_train, test_df)

        code = """
df_train = train[[id_col, target_col, 'feat_a']].copy()
df_test = test[[id_col, 'feat_a']].copy()
"""
        exec(code, g)  # noqa: S102
        assert "df_train" in g
        assert "df_test" in g
        assert list(g["df_train"].columns) == ["row_id", "target", "feat_a"]

    def test_pandas_numpy_available(self, simple_train):
        """pd и np доступны без импорта внутри кода."""
        test_df = simple_train.drop(columns=["target"])
        g = self._make_globals(simple_train, test_df)

        code = """
feat = pd.Series(np.zeros(len(train)))
df_train = train[[id_col, target_col]].copy()
df_train['zeros'] = feat.values
df_test = test[[id_col]].copy()
df_test['zeros'] = np.zeros(len(test))
"""
        exec(code, g)  # noqa: S102
        assert g["df_train"]["zeros"].sum() == 0

    def test_no_data_dir_needed(self, simple_train):
        """Если data_dir передан, NameError не возникает."""
        test_df = simple_train.drop(columns=["target"])
        g = self._make_globals(simple_train, test_df)

        code = "path = data_dir + 'orders.csv'"
        exec(code, g)  # noqa: S102
        assert g["path"] == "data/orders.csv"


# ---------------------------------------------------------------------------
# compute_stats
# ---------------------------------------------------------------------------

class TestComputeStats:
    def test_returns_features_key(self, simple_train):
        result = compute_stats(simple_train, target_col="target", id_col="row_id")
        assert "features" in result
        assert set(result["features"].keys()) == {"feat_a", "feat_b"}

    def test_stat_keys_present(self, simple_train):
        stats = compute_stats(simple_train, target_col="target", id_col="row_id")
        for feat_stats in stats["features"].values():
            for key in ("pearson", "spearman", "mutual_info", "null_pct", "nunique"):
                assert key in feat_stats, f"Отсутствует ключ '{key}'"

    def test_no_nan_in_stats(self, simple_train):
        stats = compute_stats(simple_train, target_col="target", id_col="row_id")
        for name, feat_stats in stats["features"].items():
            for key, val in feat_stats.items():
                if isinstance(val, float):
                    assert not np.isnan(val), f"{name}.{key} = NaN"

    def test_collinearity_flag(self, simple_train):
        """Две идентичные колонки должны дать high_collinearity=True."""
        df = simple_train.copy()
        df["feat_c"] = df["feat_a"]  # полная копия — коллинеарность 1.0
        stats = compute_stats(df, target_col="target", id_col="row_id")
        assert stats["features"]["feat_a"]["high_collinearity"] is True
        assert stats["features"]["feat_c"]["high_collinearity"] is True

    def test_empty_features(self):
        """Если признаков нет — возвращаем пустой dict."""
        df = pd.DataFrame({"row_id": [1, 2], "target": [0, 1]})
        result = compute_stats(df, target_col="target", id_col="row_id")
        assert result == {"features": {}}

    def test_null_pct_correct(self):
        """null_pct должен правильно считаться."""
        df = pd.DataFrame({
            "row_id": range(10),
            "feat": [1.0] * 5 + [None] * 5,
            "target": [0, 1] * 5,
        })
        stats = compute_stats(df, target_col="target", id_col="row_id")
        assert stats["features"]["feat"]["null_pct"] == 50.0


# ---------------------------------------------------------------------------
# _safe_json (через analyst_tools)
# ---------------------------------------------------------------------------

class TestSafeJson:
    def test_nan_replaced_with_null(self):
        import json
        from src.tools.analyst_tools import _safe_json

        result = _safe_json({"a": float("nan"), "b": 1.0})
        parsed = json.loads(result)
        assert parsed["a"] is None
        assert parsed["b"] == 1.0

    def test_inf_replaced_with_null(self):
        import json
        from src.tools.analyst_tools import _safe_json

        result = _safe_json({"pos": float("inf"), "neg": float("-inf")})
        parsed = json.loads(result)
        assert parsed["pos"] is None
        assert parsed["neg"] is None

    def test_nested_nan(self):
        import json
        from src.tools.analyst_tools import _safe_json

        result = _safe_json({"rows": [{"v": float("nan")}, {"v": 42.0}]})
        parsed = json.loads(result)
        assert parsed["rows"][0]["v"] is None
        assert parsed["rows"][1]["v"] == 42.0

    def test_valid_json_standard(self):
        """Результат должен парситься стандартным json.loads без ошибок."""
        import json
        from src.tools.analyst_tools import _safe_json

        data = {"x": float("nan"), "y": [1, float("inf"), 3]}
        raw = _safe_json(data)
        parsed = json.loads(raw)  # не должно упасть
        assert parsed is not None


# ---------------------------------------------------------------------------
# _get_table_columns — обработка разных форматов tables из analyst_report
# ---------------------------------------------------------------------------

class TestGetTableColumns:
    def test_tables_as_dict(self, tmp_path):
        """tables — словарь {имя: {separator: ...}} — должен работать."""
        csv = tmp_path / "foo.csv"
        csv.write_text("a,b,c\n1,2,3\n")
        report = {"tables": {"foo.csv": {"separator": ","}}}
        result = _get_table_columns(str(tmp_path) + "/", report)
        assert result.get("foo.csv") == ["a", "b", "c"]

    def test_tables_as_list(self, tmp_path):
        """tables — список имён (как возвращает GigaChat) — не должен падать."""
        csv = tmp_path / "bar.csv"
        csv.write_text("x,y\n1,2\n")
        report = {"tables": ["bar.csv", "missing.csv"]}
        result = _get_table_columns(str(tmp_path) + "/", report)
        assert result.get("bar.csv") == ["x", "y"]
        assert "missing.csv" not in result  # файла нет — тихо пропускаем

    def test_tables_missing_key(self, tmp_path):
        """Если tables вообще нет в отчёте — возвращаем пустой dict."""
        result = _get_table_columns(str(tmp_path) + "/", {})
        assert result == {}
