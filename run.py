"""Точка входа для проверки и платформы: только `python run.py`."""
from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
os.chdir(ROOT)

# Работает и без editable install (zip без uv build)
_SRC = ROOT / "src"
if _SRC.is_dir() and str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from dotenv import load_dotenv

load_dotenv(ROOT / ".env")

from mas.pipeline.submission import main


if __name__ == "__main__":
    raise SystemExit(main())
