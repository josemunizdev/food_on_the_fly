"""Raw-to-processed data pipeline entrypoint.

Loads the Zomato delivery dataset from data/raw/, validates it,
splits into train/val/test, and writes outputs to data/processed/.

Original author: Imran Ahmed
Dataset: saurabhbadole/zomato-delivery-operations-analytics-dataset
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from food_on_the_fly.config import PROCESSED_DATA_DIR, RAW_DATA_DIR
from food_on_the_fly.logging_config import get_logger, setup_logging

logger = get_logger(__name__)

RAW_FILENAME = "Zomato Dataset.csv"
RANDOM_STATE = 42


def process_data(input_dir: Path, output_dir: Path) -> None:
    """Load the raw Zomato CSV, split into train/val/test, and write to output_dir.

    Produces three files in output_dir:
        - train.csv (~80% of rows)
        - val.csv   (~10% of rows)
        - test.csv  (~10% of rows)
    """
    raw_path = input_dir / RAW_FILENAME

    if not raw_path.exists():
        logger.error("Raw dataset not found at %s", raw_path)
        logger.error("Copy '%s' from your Downloads into: %s/", RAW_FILENAME, input_dir)
        sys.exit(1)

    logger.info("Reading raw data from %s", raw_path)
    df = pd.read_csv(raw_path)
    logger.info("Loaded %d rows, %d columns", df.shape[0], df.shape[1])

    output_dir.mkdir(parents=True, exist_ok=True)

    # 80/10/10 split: first carve off the test set, then split the rest into train/val.
    # 0.111 of the remaining 90% gives ~10% val of the original.
    train_val, test = train_test_split(df, test_size=0.10, random_state=RANDOM_STATE)
    train, val = train_test_split(train_val, test_size=0.111, random_state=RANDOM_STATE)

    train.to_csv(output_dir / "train.csv", index=False)
    val.to_csv(output_dir / "val.csv", index=False)
    test.to_csv(output_dir / "test.csv", index=False)

    logger.info(
        "Wrote splits to %s: train=%d, val=%d, test=%d",
        output_dir,
        len(train),
        len(val),
        len(test),
    )


def main() -> None:
    """CLI entrypoint for data processing."""
    parser = argparse.ArgumentParser(description="Process raw Zomato data into model inputs")
    parser.add_argument("--input", type=Path, default=RAW_DATA_DIR)
    parser.add_argument("--output", type=Path, default=PROCESSED_DATA_DIR)
    args = parser.parse_args()

    setup_logging()
    process_data(args.input, args.output)
    logger.info("Data processing complete")


if __name__ == "__main__":
    main()