"""Temporal data splitting for MLOps drift detection.
Splits the dataset chronologically:
- First N days: train/val/test (baseline)
- Subsequent weeks: production data for drift detection
This simulates a realistic MLOps scenario where you
train on historical
data and monitor production data for distribution
shifts.
"""

from __future__ import annotations

from pathlib import Path

import hydra
import pandas as pd
from omegaconf import DictConfig

from food_on_the_fly.config import PROCESSED_DATA_DIR
from food_on_the_fly.data.loaders import load_dataset
from food_on_the_fly.logging_config import get_logger, setup_logging

logger = get_logger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[3]
CONFIG_DIR = str(PROJECT_ROOT / "configs")


def temporal_train_test_split(
    df: pd.DataFrame,
    date_column: str,
    baseline_days: int = 40,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split data temporally into baseline (train/val/test) and production sets.

    Args:
        df: Input dataframe
        date_column: Name of the date column
        baseline_days: Number of days for baseline (train/val/test)
        train_ratio: Proportion of baseline for training
        val_ratio: Proportion of baseline for validation
        test_ratio: Proportion of baseline for testing

    Returns:
        Tuple of (train, val, test, production) dataframes
    """
    # Parse dates and sort chronologically
    df = df.copy()
    df["_parsed_date"] = pd.to_datetime(df[date_column], format="%d-%m-%Y")
    df = df.sort_values("_parsed_date").reset_index(drop=True)

    # Find the date cutoff for baseline
    min_date = df["_parsed_date"].min()
    baseline_cutoff = min_date + pd.Timedelta(days=baseline_days)

    # Split into baseline and production
    baseline = df[df["_parsed_date"] < baseline_cutoff].copy()
    production = df[df["_parsed_date"] >= baseline_cutoff].copy()

    logger.info(
        "Baseline period: %s to %s (%d days, %d samples)",
        min_date.date(),
        baseline_cutoff.date(),
        baseline_days,
        len(baseline),
    )
    logger.info(
        "Production period: %s to %s (%d samples)",
        baseline_cutoff.date(),
        df["_parsed_date"].max().date(),
        len(production),
    )

    # Split baseline into train/val/test chronologically
    n_baseline = len(baseline)
    train_end = int(n_baseline * train_ratio)
    val_end = train_end + int(n_baseline * val_ratio)

    train = baseline.iloc[:train_end].copy()
    val = baseline.iloc[train_end:val_end].copy()
    test = baseline.iloc[val_end:].copy()

    # Drop the parsed date column (keep original format)
    train = train.drop("_parsed_date", axis=1)
    val = val.drop("_parsed_date", axis=1)
    test = test.drop("_parsed_date", axis=1)
    production = production.drop("_parsed_date", axis=1)

    logger.info(
        "Baseline splits: train=%d (%.1f%%), val=%d (%.1f%%), test=%d(%.1f%%)",
        len(train),
        100 * train_ratio,
        len(val),
        100 * val_ratio,
        len(test),
        100 * test_ratio,
    )
    return train, val, test, production


def split_production_by_weeks(
    df: pd.DataFrame, date_column: str, window_days: int = 7
) -> dict[str, pd.DataFrame]:
    """Split production data into weekly windows.

    Args:
      df: Production dataframe
      date_column: Name of the date column
      window_days: Size of each window in days

    Returns:
        Dictionary mapping week names to dataframes
    """
    if len(df) == 0:
        logger.warning("No production data to split into weeks")
        return {}

    # Parse dates temporarily for splitting
    df = df.copy()
    df["_parsed_date"] = pd.to_datetime(df[date_column], format="%d-%m-%Y")

    min_date = df["_parsed_date"].min()
    max_date = df["_parsed_date"].max()

    weeks = {}
    week_num = 1
    current_start = min_date

    while current_start <= max_date:
        current_end = current_start + pd.Timedelta(days=window_days)
        week_data = df[
            (df["_parsed_date"] >= current_start) & (df["_parsed_date"] < current_end)
        ].copy()

        if len(week_data) > 0:
            # Drop temp column before saving
            week_data = week_data.drop("_parsed_date", axis=1)
            week_name = f"week{week_num}"
            weeks[week_name] = week_data
            logger.info(
                "Week %d: %s to %s (%d samples)",
                week_num,
                current_start.date(),
                current_end.date(),
                len(week_data),
            )

        current_start = current_end
        week_num += 1

    return weeks


@hydra.main(version_base=None, config_path=CONFIG_DIR, config_name="config")
def main(cfg: DictConfig) -> None:
    """Create temporal splits of the dataset for MLOps drift detection."""
    setup_logging()

    output_dir = PROCESSED_DATA_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 80)
    logger.info("TEMPORAL DATA SPLIT FOR MLOPS")
    logger.info("=" * 80)

    # Load dataset
    logger.info("Loading dataset from %s", cfg.data.source_uri)
    df = load_dataset(cfg.data.source_uri)
    logger.info("Loaded %d rows, %d columns", df.shape[0], df.shape[1])

    # Get temporal split config
    baseline_days = cfg.data.get("temporal_baseline_days", 40)
    window_days = cfg.data.get("temporal_window_days", 7)
    train_ratio = cfg.data.get("temporal_train_ratio", 0.7)
    val_ratio = cfg.data.get("temporal_val_ratio", 0.15)
    test_ratio = cfg.data.get("temporal_test_ratio", 0.15)

    # Create temporal splits
    train, val, test, production = temporal_train_test_split(
        df,
        date_column="Order_Date",
        baseline_days=baseline_days,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        test_ratio=test_ratio,
    )

    # Save baseline splits
    train.to_csv(output_dir / "train.csv", index=False)
    val.to_csv(output_dir / "val.csv", index=False)
    test.to_csv(output_dir / "test.csv", index=False)

    logger.info("Saved baseline splits to %s", output_dir)

    # Split production data into weeks
    if len(production) > 0:
        weeks = split_production_by_weeks(production, "Order_Date", window_days)

        for week_name, week_data in weeks.items():
            week_path = output_dir / f"{week_name}.csv"
            week_data.to_csv(week_path, index=False)
            logger.info("Saved %s: %d samples", week_path.name, len(week_data))

        logger.info("Saved %d weekly splits for drift detection", len(weeks))
    else:
        logger.warning("No production data available for weekly splits")

    logger.info("=" * 80)
    logger.info("TEMPORAL SPLIT COMPLETE")
    logger.info("=" * 80)
    logger.info("Next steps:")
    logger.info("1. Train model: make train")
    logger.info("2. Analyze drift: python -m food_on_the_fly.analysis.drift_detection")


if __name__ == "__main__":
    main()
