from __future__ import annotations

import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import pandas as pd

# Добавляем корень проекта в sys.path чтобы импорты работали при любом cwd
_project_root = str(Path(__file__).resolve().parents[2])
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from src.utils.scoring import ScoringEngine

try:
    import tomllib  # py3.11+
except ModuleNotFoundError:  # pragma: no cover
    import tomli as tomllib  # type: ignore


ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data"
OUTPUT_DIR = ROOT / "output"
PYPROJECT_PATH = ROOT / "pyproject.toml"
RUN_PATH = ROOT / "run.py"
ENV_PATH = ROOT / ".env"
HIDDEN_LABELS_PATH = ROOT / "configs" / "test.csv"

MAX_RUNTIME_SEC = 600
MAX_FEATURES = 5


def read_table(path: Path) -> pd.DataFrame:
    """
    Читает csv с автоопределением разделителя через csv.Sniffer.
    sep=None + engine=python ломается на однoколоночных файлах.
    """
    import csv
    assert path.exists(), f"Файл не найден: {path}"
    with path.open(newline="", encoding="utf-8") as f:
        sample = f.read(4096)
    try:
        dialect = csv.Sniffer().sniff(sample, delimiters=",;\t|")
        sep = dialect.delimiter
    except csv.Error:
        sep = ","
    return pd.read_csv(path, sep=sep)


def load_pyproject() -> dict:
    assert PYPROJECT_PATH.exists(), "Отсутствует pyproject.toml"
    with PYPROJECT_PATH.open("rb") as f:
        return tomllib.load(f)


def get_project_dependencies(pyproject: dict) -> list[str]:
    project = pyproject.get("project", {})
    deps = project.get("dependencies", [])
    assert isinstance(deps, list), "project.dependencies в pyproject.toml должен быть списком"
    return [str(x).lower() for x in deps]


def ensure_env_file() -> None:
    assert ENV_PATH.exists(), "Отсутствует .env"

    content = ENV_PATH.read_text(encoding="utf-8").split('\n')
    assert any(env_var.startswith("GIGACHAT_CREDENTIALS") for env_var in content), "В .env отсутствует GIGACHAT_CREDENTIALS"
    assert any(env_var.startswith("GIGACHAT_SCOPE") for env_var in content), "В .env отсутствует GIGACHAT_SCOPE"

def ensure_required_files() -> None:
    assert RUN_PATH.exists(), "Отсутствует run.py"
    assert DATA_DIR.exists() and DATA_DIR.is_dir(), "Отсутствует папка data/"
    assert (DATA_DIR / "train.csv").exists(), "Отсутствует data/train.csv (нужно лишь для запуска локальной проверки)"
    assert (DATA_DIR / "test.csv").exists(), "Отсутствует data/test.csv (нужно лишь для запуска локальной проверки)"
    assert (DATA_DIR / "readme.txt").exists(), "Отсутствует data/readme.txt (нужно лишь для запуска локальной проверки)"


def ensure_dependencies() -> None:
    pyproject = load_pyproject()
    deps = get_project_dependencies(pyproject)

    required_markers = ["catboost", "pandas", "numpy",
                        "langchain-gigachat", "python-dotenv"]
    missing = [dep for dep in required_markers if not any(dep in x for x in deps)]

    assert not missing, (
        "В pyproject.toml отсутствуют обязательные зависимости "
        f"(или их нельзя однозначно распознать): {missing}"
    )


def clean_output_dir() -> None:
    if OUTPUT_DIR.exists():
        shutil.rmtree(OUTPUT_DIR)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def run_solution() -> tuple[int, float, str, str]:
    env = os.environ.copy()
    log_path = ROOT / "run_agent.log"

    start = time.perf_counter()
    with open(log_path, "w", encoding="utf-8") as log_file:
        proc = subprocess.run(
            [sys.executable, "run.py"],
            cwd=ROOT,
            stdout=log_file,
            stderr=log_file,
            timeout=MAX_RUNTIME_SEC,
            env=env,
        )
    elapsed = time.perf_counter() - start

    log_text = log_path.read_text(encoding="utf-8", errors="replace")
    return proc.returncode, elapsed, log_text, ""


def assert_output_files_exist() -> tuple[Path, Path]:
    train_out = OUTPUT_DIR / "train.csv"
    test_out = OUTPUT_DIR / "test.csv"

    assert train_out.exists(), "После запуска не найден output/train.csv"
    assert test_out.exists(), "После запуска не найден output/test.csv"

    assert train_out.stat().st_size > 0, "output/train.csv пустой"
    assert test_out.stat().st_size > 0, "output/test.csv пустой"

    return train_out, test_out


def assert_output_structure(
    input_train: pd.DataFrame,
    input_test: pd.DataFrame,
    output_train: pd.DataFrame,
    output_test: pd.DataFrame,
) -> None:

    print(input_test)

    # 1. Проверка обязательных колонок
    for col in input_train.columns:
        assert col in output_train.columns, f"В output/train.csv отсутствует колонка: {col}"
    for col in input_test.columns:
        assert col in output_test.columns, f"В output/test.csv отсутствует колонка: {col}"

    # 2. Проверка фичей
    reserved_train = set(input_train.columns)
    reserved_test = set(input_test.columns)

    feature_cols_train = [c for c in output_train.columns if c not in reserved_train]
    feature_cols_test = [c for c in output_test.columns if c not in reserved_test]

    assert feature_cols_train == feature_cols_test, (
        "Набор признаков в output/train.csv и output/test.csv должен совпадать по именам и порядку.\n"
        f"train features: {feature_cols_train}\n"
        f"test features: {feature_cols_test}"
    )

    assert 1 <= len(feature_cols_train) <= MAX_FEATURES, (
        f"Количество признаков должно быть от 1 до {MAX_FEATURES}, "
        f"получено: {len(feature_cols_train)}"
    )

    # 3. Проверка, что признаки содержат данные
    for col in feature_cols_train:
        assert not output_train[col].isna().all(), f"Признак {col} в train полностью NaN"
        assert not output_test[col].isna().all(), f"Признак {col} в test полностью NaN"

    # 4. Проверка на дубли имен колонок
    assert output_train.columns.is_unique, "В output/train.csv есть дублирующиеся имена колонок"
    assert output_test.columns.is_unique, "В output/test.csv есть дублирующиеся имена колонок"


def main() -> None:
    ensure_required_files()
    ensure_env_file()
    ensure_dependencies()

    input_train = read_table(DATA_DIR / "train.csv")
    input_test = read_table(DATA_DIR / "test.csv")

    #clean_output_dir()

    try:
        #returncode, elapsed, log_output, _ = run_solution()
        returncode, elapsed, log_output, = 0, 0, []
    except subprocess.TimeoutExpired as e:
        log_path = ROOT / "run_agent.log"
        tail = log_path.read_text(encoding="utf-8", errors="replace")[-3000:] if log_path.exists() else ""
        raise AssertionError(
            f"Решение превысило лимит времени {MAX_RUNTIME_SEC} секунд\n\nПоследние логи:\n{tail}"
        ) from e

    print("\n--- Лог агента ---")
    print(log_output[-8000:] if len(log_output) > 8000 else log_output)
    print("--- Конец лога ---\n")

    assert returncode == 0, (
        f"run.py завершился с ошибкой (код {returncode}). "
        "Лог выведен выше."
    )

    assert elapsed <= MAX_RUNTIME_SEC, (
        f"Решение работало слишком долго: {elapsed:.2f} сек. "
        f"Лимит: {MAX_RUNTIME_SEC} сек."
    )

    train_out_path, test_out_path = assert_output_files_exist()

    output_train = read_table(train_out_path)
    output_test = read_table(test_out_path)

    assert_output_structure(
        input_train=input_train,
        input_test=input_test,
        output_train=output_train,
        output_test=output_test,
    )

    agent_elapsed = elapsed
    n_features = len(output_test.columns) - len(input_test.columns)

    print("OK: submit passed basic checks")
    print(f"Generated features : {n_features}")
    print(f"Output files       : {train_out_path}, {test_out_path}")
    print(f"Agent runtime      : {agent_elapsed:.2f} sec")

    id_col = input_train.columns[0]
    # Ищем target явно: последняя колонка input_train, или колонка с именем "target"
    target_candidates = [c for c in input_train.columns if c.lower() == "target"]
    target_col = target_candidates[0] if target_candidates else input_train.columns[-1]

    # Оригинальные колонки input — не признаки
    original_cols = set(input_train.columns) | set(input_test.columns)

    print("\nЗапуск скоринга (CatBoost 5-fold CV + test ROC-AUC)...")
    sc = ScoringEngine(
        id_column=id_col,
        target_column=target_col,
        original_columns=original_cols,
        hidden_labels_path=HIDDEN_LABELS_PATH if HIDDEN_LABELS_PATH.exists() else None,
    )
    result = sc.score(str(OUTPUT_DIR))
    catboost_elapsed = result.scoring_elapsed
    total_elapsed = agent_elapsed + catboost_elapsed

    print()
    print("=" * 45)
    print(f"  CV ROC-AUC (train)  : {result.cv_roc_auc:.4f} ± {result.cv_std:.4f}")
    print(f"  CV folds            : {result.cv_folds}")
    if result.test_roc_auc is not None:
        print(f"  Test ROC-AUC        : {result.test_roc_auc:.4f}")
        print(f"  Test Gini           : {result.test_gini:.4f}")
    else:
        print("  Test ROC-AUC        : N/A (configs/test.csv не найден)")
    print(f"  Features            : {list(result.top_features.keys())}")
    print("=" * 45)
    print(f"  Agent time          : {agent_elapsed:.2f} sec")
    print(f"  CatBoost time       : {catboost_elapsed:.2f} sec")
    print(f"  Total time          : {total_elapsed:.2f} sec")
    print("=" * 45)
    


if __name__ == "__main__":
    main()