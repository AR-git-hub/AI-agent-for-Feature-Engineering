"""EDA и подготовка таблиц — тулы первого агента (DatasetAgent)."""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

from src.mas.domain.models import DatetimeColumnStats, EDASummary, PrepareDataReport


def summarize_column(df: pd.DataFrame, column: str) -> EDASummary:
    s = df[column]
    null_ratio = float(s.isna().mean()) if len(df) else 0.0
    dtype = str(s.dtype)
    notes_parts: list[str] = []
    if pd.api.types.is_numeric_dtype(s):
        if s.notna().any():
            notes_parts.append(f"mean={s.mean():.4g}")
            if s.notna().sum() > 1:
                notes_parts.append(f"std={s.std():.4g}")
            else:
                notes_parts.append("std=N/A (one value)")
    else:
        vc = s.astype(str).value_counts(dropna=True).head(5)
        notes_parts.append("top_values=" + ";".join(f"{i}:{v}" for i, v in vc.items()))
    return EDASummary(
        column=column,
        dtype=dtype,
        non_null_count=int(s.notna().sum()),
        null_ratio=null_ratio,
        notes=", ".join(notes_parts),
    )


def eda_all_columns(df: pd.DataFrame) -> dict[str, EDASummary]:
    return {c: summarize_column(df, c) for c in df.columns}


def merge(table1: pd.DataFrame, table2: pd.DataFrame, drop_similar: bool = False) -> pd.DataFrame:
    res = pd.concat([table1, table2], axis=1, join="outer")

    duplicate_cols = res.columns[res.columns.duplicated(keep=False)].unique()
    for col in duplicate_cols:
        col_group = res.loc[:, res.columns == col]
        merged_col = col_group.iloc[:, 0]
        for i in range(1, col_group.shape[1]):
            merged_col = merged_col.combine_first(col_group.iloc[:, i])
        res = res.loc[:, res.columns != col]
        res[col] = merged_col

    if drop_similar:
        cols = res.columns.tolist()
        cols_to_drop: set[str] = set()

        for i, col_a in enumerate(cols):
            if col_a in cols_to_drop:
                continue
            for col_b in cols[i + 1 :]:
                if col_b in cols_to_drop:
                    continue
                if res[col_a].equals(res[col_b]):
                    cols_to_drop.add(col_b)

        res = res.drop(columns=list(cols_to_drop))

    return res


def basic_eda(data: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for col in data.columns:
        s = data[col]
        total = len(s)
        non_null = int(s.notna().sum())
        null_count = total - non_null

        row: dict = {
            "column": col,
            "dtype": str(s.dtype),
            "total": total,
            "non_null": non_null,
            "null_count": null_count,
            "null_ratio": round(null_count / total, 4) if total else 0.0,
            "unique_count": s.nunique(dropna=True),
            "unique_ratio": round(s.nunique(dropna=True) / non_null, 4) if non_null else 0.0,
        }

        if pd.api.types.is_numeric_dtype(s):
            valid = s.dropna()
            row |= {
                "mean": valid.mean() if non_null > 0 else None,
                "std": valid.std() if non_null > 1 else None,
                "min": valid.min() if non_null > 0 else None,
                "q25": valid.quantile(0.25) if non_null > 0 else None,
                "median": valid.median() if non_null > 0 else None,
                "q75": valid.quantile(0.75) if non_null > 0 else None,
                "max": valid.max() if non_null > 0 else None,
                "skewness": valid.skew() if non_null > 2 else None,
                "kurtosis": valid.kurtosis() if non_null > 3 else None,
                "outlier_count": _count_outliers(valid) if non_null > 0 else None,
                "mode": None,
                "top5": None,
            }
        else:
            vc = s.astype(str).value_counts(dropna=True)
            top5 = "; ".join(f"{v!r}:{c}" for v, c in vc.head(5).items())
            row |= {
                "mean": None,
                "std": None,
                "min": None,
                "q25": None,
                "median": None,
                "q75": None,
                "max": None,
                "skewness": None,
                "kurtosis": None,
                "outlier_count": None,
                "mode": vc.index[0] if len(vc) else None,
                "top5": top5,
            }

        rows.append(row)

    return pd.DataFrame(rows).set_index("column")


def _count_outliers(s: pd.Series) -> int:
    q1, q3 = s.quantile(0.25), s.quantile(0.75)
    iqr = q3 - q1
    return int(((s < q1 - 1.5 * iqr) | (s > q3 + 1.5 * iqr)).sum())


def _analyze_class_balance(data: pd.DataFrame, target_col: str) -> pd.DataFrame:
    vc = data[target_col].value_counts(dropna=False)
    total = len(data)
    result = pd.DataFrame(
        {
            "count": vc,
            "ratio": vc / total,
        }
    )
    result.index.name = target_col
    return result


def _analyze_distributions(data: pd.DataFrame, target_col: str | None = None) -> pd.DataFrame:
    rows = []

    numeric_cols = data.select_dtypes(include="number").columns.tolist()
    if target_col in numeric_cols:
        numeric_cols.remove(target_col)

    cat_cols = data.select_dtypes(exclude="number").columns.tolist()
    if target_col in cat_cols:
        cat_cols.remove(target_col)

    for col in numeric_cols:
        s = data[col].dropna()
        row: dict = {"column": col, "kind": "numeric"}

        row["skewness"] = s.skew()
        row["kurtosis"] = s.kurtosis()

        if len(s) >= 8:
            if len(s) <= 5000:
                _, p_value = stats.shapiro(s)
            else:
                _, p_value = stats.normaltest(s)
            row["normality_p"] = round(p_value, 4)
            row["is_normal"] = p_value > 0.05
        else:
            row["normality_p"] = None
            row["is_normal"] = None

        if len(s) < 2 or s.std() == 0:
            row["peaks"] = 1
            row["is_multimodal"] = False
        else:
            kde = stats.gaussian_kde(s)
            x = np.linspace(s.min(), s.max(), 200)
            density = kde(x)
            peaks = ((density[1:-1] > density[:-2]) & (density[1:-1] > density[2:])).sum()
            row["peaks"] = int(peaks)
            row["is_multimodal"] = peaks > 1

        if target_col is not None:
            for cls in data[target_col].dropna().unique():
                sub = data.loc[data[target_col] == cls, col].dropna()
                row[f"mean_class_{cls}"] = sub.mean()
                row[f"std_class_{cls}"] = sub.std() if len(sub) > 1 else None

        rows.append(row)

    for col in cat_cols:
        s = data[col].dropna().astype(str)
        probs = s.value_counts(normalize=True)
        entropy = float(stats.entropy(probs))
        row = {
            "column": col,
            "kind": "categorical",
            "entropy": round(entropy, 4),
            "is_multimodal": None,
            "peaks": None,
            "skewness": None,
            "kurtosis": None,
            "normality_p": None,
            "is_normal": None,
        }
        rows.append(row)

    if not rows:
        return pd.DataFrame()

    return pd.DataFrame(rows).set_index("column")


def _analyze_datetime_columns(data: pd.DataFrame) -> dict[str, DatetimeColumnStats]:
    result: dict[str, DatetimeColumnStats] = {}
    dt_cols = data.select_dtypes(include=["datetime64"]).columns

    for col in dt_cols:
        s = data[col].dropna()
        counts_by_month = s.dt.month.value_counts().sort_index().rename("count_by_month")
        counts_by_weekday = s.dt.dayofweek.value_counts().sort_index().rename("count_by_weekday")

        ts = s.sort_values()
        x = np.arange(len(ts))
        y = ts.astype(np.int64)
        slope, _, r_value, p_value, _ = stats.linregress(x, y)

        result[col] = DatetimeColumnStats(
            by_month=counts_by_month.to_frame(),
            by_weekday=counts_by_weekday.to_frame(),
            trend_slope=float(slope),
            trend_r2=float(r_value**2),
            trend_p=float(p_value),
            has_trend=bool(p_value < 0.05),
        )

    return result


def _drop_duplicates(
    data: pd.DataFrame,
    key_col: str | None = None,
    keep: str = "first",
) -> tuple[pd.DataFrame, int]:
    before = len(data)

    if key_col:
        if key_col in data.columns:
            res = data.drop_duplicates(subset=[key_col], keep=keep)
        elif data.index.name == key_col:
            res = data[~data.index.duplicated(keep=keep)]
        else:
            raise KeyError(f"'{key_col}' не найден ни среди колонок, ни в индексе")
    else:
        res = data.drop_duplicates(keep=keep)

    return res, before - len(res)


def _fix_dtypes(data: pd.DataFrame) -> pd.DataFrame:
    res = data.copy()
    for col in res.select_dtypes(include="object").columns:
        converted = pd.to_numeric(res[col], errors="coerce")
        if converted.notna().sum() / max(len(res[col]), 1) >= 0.9:
            res[col] = converted
            continue

        try:
            import warnings
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                converted_dt = pd.to_datetime(res[col], errors="coerce")
            if converted_dt.notna().sum() / max(res[col].notna().sum(), 1) >= 0.9:
                res[col] = converted_dt
        except Exception:
            pass

    return res


def _normalize_categories(data: pd.DataFrame) -> pd.DataFrame:
    res = data.copy()
    for col in res.select_dtypes(include="object").columns:
        res[col] = res[col].str.strip().str.lower()
    return res


def _drop_constant_columns(
    data: pd.DataFrame,
    threshold: float = 0.995,
    exclude: list[str] | None = None,
) -> tuple[pd.DataFrame, list[str]]:
    exclude = exclude or []
    to_drop = []
    for col in data.columns:
        if col in exclude:
            continue
        top_ratio = data[col].value_counts(normalize=True, dropna=False).iloc[0]
        if top_ratio >= threshold:
            to_drop.append(col)
    return data.drop(columns=to_drop), to_drop


def _drop_high_cardinality(
    data: pd.DataFrame,
    threshold: float = 0.95,
    exclude: list[str] | None = None,
) -> tuple[pd.DataFrame, list[str]]:
    exclude = exclude or []
    to_drop = []
    cat_cols = data.select_dtypes(include="object").columns
    for col in cat_cols:
        if col in exclude:
            continue
        non_null = data[col].notna().sum()
        if non_null == 0:
            continue
        ratio = data[col].nunique() / non_null
        if ratio >= threshold:
            to_drop.append(col)
    return data.drop(columns=to_drop), to_drop


def prepare_data(
    data: pd.DataFrame,
    target_col: str,
    key_col: str | None = None,
    constant_threshold: float = 0.995,
    cardinality_threshold: float = 0.95,
) -> tuple[pd.DataFrame, PrepareDataReport]:
    df = _fix_dtypes(data)
    df = _normalize_categories(df)
    df, n_dupes = _drop_duplicates(df, key_col=key_col)
    df, const_cols = _drop_constant_columns(
        df,
        threshold=constant_threshold,
        exclude=[target_col],
    )
    df, hc_cols = _drop_high_cardinality(
        df,
        threshold=cardinality_threshold,
        exclude=[target_col],
    )
    report = PrepareDataReport(
        duplicates_dropped=n_dupes,
        constant_cols_dropped=const_cols,
        high_cardinality_cols_dropped=hc_cols,
        class_balance=_analyze_class_balance(df, target_col),
        distributions=_analyze_distributions(df, target_col),
        datetime_info=_analyze_datetime_columns(df),
        basic_eda=basic_eda(df),
    )
    return df, report


def _all_features(df: pd.DataFrame) -> list:
    cols = list(df.columns)
    if df.index.name and df.index.name not in cols:
        cols = [df.index.name] + cols
    return cols


def _ensure_column(df: pd.DataFrame, col: str) -> None:
    if col in df.columns:
        return

    if isinstance(df.index, pd.MultiIndex):
        if col in df.index.names:
            df.reset_index(inplace=True)
    else:
        if df.index.name == col:
            df.reset_index(inplace=True)


def merge_through_key(
    table1: pd.DataFrame,
    table2: pd.DataFrame,
    key: str | None = None,
) -> pd.DataFrame:
    try:
        t1 = table1.copy()
        t2 = table2.copy()

        if key is None:
            features2 = set(_all_features(t2))
            common = [f for f in _all_features(t1) if f in features2]

            if not common:
                return table1.copy()

            if len(common) > 1:
                key = min(
                    common,
                    key=lambda c: t2[c].duplicated().sum() if c in t2.columns else np.inf,
                )
            else:
                key = common[0]

        _ensure_column(t1, key)
        _ensure_column(t2, key)

        if key not in t1.columns or key not in t2.columns:
            return table1.copy()

        if t1[key].dtype != t2[key].dtype:
            try:
                t2[key] = t2[key].astype(t1[key].dtype)
            except Exception:
                pass

        orig_index = t1.index

        _OI = "__orig_idx__"
        while _OI in t1.columns:
            _OI += "_"

        t1[_OI] = np.arange(len(t1))

        t2["_order"] = np.arange(len(t2))
        t2 = t2.sort_values("_order")
        t2_dedup = t2.drop_duplicates(subset=[key], keep="first")
        t2_dedup = t2_dedup.drop(columns=["_order"])

        result = t1.merge(t2_dedup, on=key, how="left", suffixes=("_x", "_y"))

        result.sort_values(_OI, kind="stable", inplace=True)
        result.drop(columns=[_OI], inplace=True)

        result.index = orig_index

        x_cols = [c for c in result.columns if c.endswith("_x")]
        rename_map: dict[str, str] = {}
        drop_cols: list[str] = []

        for cx in x_cols:
            base = cx[:-2]
            cy = base + "_y"
            if cy not in result.columns:
                continue

            try:
                identical = (
                    result[cx].dtype == result[cy].dtype
                    and result[cx].equals(result[cy])
                )
            except Exception:
                identical = False

            if identical:
                rename_map[cx] = base
                drop_cols.append(cy)
            else:
                rename_map[cx] = base + "_1"
                rename_map[cy] = base + "_2"

        if drop_cols:
            result.drop(columns=drop_cols, inplace=True)

        if rename_map:
            result.rename(columns=rename_map, inplace=True)

        return result

    except Exception:
        return table1.copy()
