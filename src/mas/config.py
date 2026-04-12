import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
# Боевой / обучающий ввод по ТЗ соревнования — только из `data/`.
DATA_DIR = Path(os.environ.get("DATA_DIR", PROJECT_ROOT / "data"))
OUTPUT_DIR = Path(os.environ.get("OUTPUT_DIR", PROJECT_ROOT / "output"))
# Мини-датасеты и сценарии для pytest — из `configs/` (основной пайплайн сюда не ходит).
CONFIGS_DIR = Path(os.environ.get("CONFIGS_DIR", PROJECT_ROOT / "configs"))
