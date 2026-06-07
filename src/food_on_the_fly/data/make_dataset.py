"""Raw-to-processed data pipeline entrypoint.

Reads the food delivery dataset from the GCS URI defined in
configs/config.yaml (data.source_uri), splits it into train/val/test,
and writes the splits to data/processed/ for use by train_model.py.

Run from the repo root with:
    make data
    # or:
    python -m food_on_the_fly.data.make_dataset
"""

from __future__ import annotations

from pathlib import Path

import hydra
import pandas as pd
from omegaconf import DictConfig
from sklearn.model_selection import train_test_split

from food_on_the_fly.config import PROCESSED_DATA_DIR
from food_on_the_fly.data.loaders import load_dataset
from food_on_the_fly.logging_config import get_logger, setup_logging

logger = get_logger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[3]
CONFIG_DIR = str(PROJECT_ROOT / "configs")


def split_by_date(
    df: pd.DataFrame,
    date_col: str = "Order_Date",
    baseline_days: int = 40,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Split data by strict temporal cutoff.

    Args:
        df: DataFrame with date column
        date_col: Name of date column
        baseline_days: Number of days for modeling window
        train_ratio: Fraction of baseline for training
        val_ratio: Fraction of baseline for validation
        test_ratio: Fraction of baseline for testing

    Returns:
        train, val, test, drift (test2)

    Ensures:
        - max(train|val|test) < min(drift)
        - Every row in exactly one split
        - No data leakage
    """
    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col], format="%d-%m-%Y")

    # find boundaries
    min_date = df[date_col].min()
    cutoff_date = min_date + pd.Timedelta(days=baseline_days)

    # split data
    modeling = df[df[date_col] < cutoff_date].copy()
    drift = df[df[date_col] >= cutoff_date].copy()

    # further split baseline into train/val/test

    train, val_test = train_test_split(
        modeling, test_size=(val_ratio + test_ratio), random_state=random_state
    )
    val, test = train_test_split(
        val_test,
        test_size=test_ratio / (val_ratio + test_ratio),
        random_state=random_state,
    )

    assert len(train) + len(val) + len(test) == len(modeling), (
        "Splits must partition modeling data"
    )
    assert len(train) + len(val) + len(test) + len(drift) == len(df), (
        "Splits must partition all data"
    )
    assert modeling[date_col].max() < drift[date_col].min(), (
        "Modeling data must be strictly before drift data"
    )
    return train, val, test, drift


@hydra.main(version_base=None, config_path=CONFIG_DIR, config_name="config")
def main(cfg: DictConfig) -> None:
    """Read the dataset from cfg.data.source_uri and write train/val/test splits."""
    setup_logging()

    output_dir = PROCESSED_DATA_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Reading dataset from %s", cfg.data.source_uri)
    df = load_dataset(cfg.data.source_uri)
    logger.info("Loaded %d rows, %d columns", df.shape[0], df.shape[1])

    test_split = 1.0 - cfg.data.train_test_split

    # First carve off the test set, then split the remainder into train/val.
    train_df, val_df, test_df, drift_df = split_by_date(
        df,
        date_col=cfg.data.date_col,
        baseline_days=cfg.data.temporal_baseline_days,
        train_ratio=1.0 - test_split,
        val_ratio=cfg.data.temporal_val_ratio,
        test_ratio=cfg.data.temporal_test_ratio,
        random_state=cfg.data.random_state,
    )
    train_path = output_dir / "train.csv"
    val_path = output_dir / "val.csv"
    test_path = output_dir / "test.csv"
    drift_path = output_dir / "drift.csv"

    train_df.to_csv(train_path, index=False)
    val_df.to_csv(val_path, index=False)
    test_df.to_csv(test_path, index=False)
    drift_df.to_csv(drift_path, index=False)

    logger.info(
        "Wrote splits to %s: train=%d, val=%d, test=%d, drift=%d",
        output_dir,
        len(train_df),
        len(val_df),
        len(test_df),
        len(drift_df),
    )
    logger.info("Data processing complete. Run `make train` next.")


if __name__ == "__main__":
    main()
