"""Compare the current run's validation metrics against the committed baseline.

Reads the freshly produced ``metrics.txt`` (written by ``train_model.py``) and
``reports/baseline_metrics.json``, then writes a Markdown comparison table to
``comparison.md`` for the CML PR comment.

Usage:
    python scripts/compare_to_baseline.py
    python scripts/compare_to_baseline.py --metrics metrics.txt \
        --baseline reports/baseline_metrics.json --out comparison.md
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def parse_metrics_txt(path: Path) -> dict[str, float]:
    """Extract validation RMSE/MAE/R2 from the train_model.py metrics.txt format."""
    text = path.read_text(encoding="utf-8")
    val_split = re.split(r"===\s*VALIDATION METRICS\s*===", text)
    if len(val_split) < 2:
        raise ValueError(
            f"Could not find a VALIDATION METRICS section in {path}. "
            "Has the metrics.txt format changed?"
        )
    val_block = val_split[1]

    def grab(label: str) -> float:
        m = re.search(rf"{label}\s*:\s*([-+]?\d*\.?\d+)", val_block)
        if not m:
            raise ValueError(
                f"Could not parse '{label}' from validation block of {path}"
            )
        return float(m.group(1))

    return {
        "val_rmse": grab("RMSE"),
        "val_mae": grab("MAE"),
        "val_r2": grab(r"R\u00b2"),
    }


def fmt_delta(delta: float, lower_is_better: bool, places: int = 3) -> str:
    sign = "+" if delta > 0 else ""
    rounded = round(delta, places)
    if abs(rounded) < (10**-places):
        return f"{sign}{rounded:.{places}f} | same"
    improved = (rounded < 0) if lower_is_better else (rounded > 0)
    flag = "better" if improved else "worse"
    return f"{sign}{rounded:.{places}f} | {flag}"


def build_table(current: dict[str, float], baseline: dict[str, float]) -> str:
    rows = [
        ("Val RMSE (min)", "val_rmse", True, 2),
        ("Val MAE (min)", "val_mae", True, 2),
        ("Val R2", "val_r2", False, 4),
    ]
    lines = [
        "### Model Comparison: Current vs. Baseline",
        "",
        f"Baseline: **{baseline.get('model', 'baseline')}** "
        f"({baseline.get('description', 'reference model')})",
        "",
        "| Metric | Baseline | Current (PR) | Delta | |",
        "|---|---|---|---|---|",
    ]
    for label, key, lower_is_better, places in rows:
        b = baseline[key]
        c = current[key]
        delta = c - b
        lines.append(
            f"| {label} | {b:.{places}f} | {c:.{places}f} | "
            f"{fmt_delta(delta, lower_is_better, places)} |"
        )
    lines += [
        "",
        "_Lower is better for RMSE/MAE; higher is better for R2. "
        "Delta is current minus baseline._",
        "",
    ]
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--metrics", default="metrics.txt")
    ap.add_argument(
        "--baseline",
        default=str(PROJECT_ROOT / "reports" / "baseline_metrics.json"),
    )
    ap.add_argument("--out", default="comparison.md")
    args = ap.parse_args()

    current = parse_metrics_txt(Path(args.metrics))
    baseline = json.loads(Path(args.baseline).read_text(encoding="utf-8"))

    table = build_table(current, baseline)
    Path(args.out).write_text(table + "\n", encoding="utf-8")
    print(table)


if __name__ == "__main__":
    main()
