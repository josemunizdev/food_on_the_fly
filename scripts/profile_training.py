"""Profile the training entrypoint with cProfile.

Outputs:
  - reports/profiling/train_model.prof  (binary, viewable with snakeviz/pstats)
  - reports/profiling/train_model.txt   (top-50 cumulative-time text report)

Usage:
    python scripts/profile_training.py
    snakeviz reports/profiling/train_model.prof   # optional visualizer
"""

from __future__ import annotations

import cProfile
import pstats
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC = PROJECT_ROOT / "src"
sys.path.insert(0, str(SRC))

OUT_DIR = PROJECT_ROOT / "reports" / "profiling"
OUT_DIR.mkdir(parents=True, exist_ok=True)
BIN_PATH = OUT_DIR / "train_model.prof"
TXT_PATH = OUT_DIR / "train_model.txt"


def _run_training() -> None:
    from food_on_the_fly.train_model import main as train_main

    train_main()


def main() -> None:
    profiler = cProfile.Profile()
    profiler.enable()
    try:
        _run_training()
    finally:
        profiler.disable()

    profiler.dump_stats(str(BIN_PATH))

    with TXT_PATH.open("w") as f:
        stats = pstats.Stats(profiler, stream=f).sort_stats("cumulative")
        stats.print_stats(50)

    print(f"Wrote profile to {BIN_PATH}")
    print(f"Wrote text report to {TXT_PATH}")
    print(f"View interactively with: snakeviz {BIN_PATH}")


if __name__ == "__main__":
    main()
