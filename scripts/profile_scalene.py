"""Profile training with Scalene (memory + CPU for sklearn/XGBoost).

Scalene is a high-performance CPU and memory profiler for Python that is
especially useful for data science workloads using sklearn/XGBoost.

Outputs:
    reports/profiling/scalene_report.html - Interactive HTML report

Usage:
    python scripts/profile_scalene.py
    open reports/profiling/scalene_report.html  # View in browser
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = PROJECT_ROOT / "reports" / "profiling"
OUT_DIR.mkdir(parents=True, exist_ok=True)
REPORT_PATH = OUT_DIR / "scalene_report.html"


def main() -> None:
    """Run Scalene profiler on train_model."""
    train_script = PROJECT_ROOT / "src" / "food_on_the_fly" / "train_model.py"
    json_output = OUT_DIR / "scalene_report.json"
    cmd = [
        "scalene",
        "run",
        "--outfile",
        str(json_output),
        "--html",
        str(train_script),
    ]

    print("Running Scalene profiler...")
    print(f"Command: {' '.join(cmd)}")

    import os

    env = os.environ.copy()
    env["PYTHONPATH"] = str(PROJECT_ROOT / "src")  # Ensure
    env["PYTHONPATH"] = str(PROJECT_ROOT / "src")  # Ensure

    result = subprocess.run(cmd, cwd=PROJECT_ROOT, env=env)

    if json_output.exists():
        print("Note: Hydra warnings are normal and don't affect profiling")
        print(f"✅ Scalene report created: {REPORT_PATH}")
        print(f"   Open with: open {REPORT_PATH}")
    else:
        print(f"❌ Scalene profiling failed with code {result.returncode}")
        sys.exit(1)


if __name__ == "__main__":
    main()
