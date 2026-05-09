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
from omegaconf import DictConfig
from sklearn.model_selection import train_test_split

from food_on_the_fly.config import PROCESSED_DATA_DIR
from food_on_the_fly.data.loaders import load_dataset
from food_on_the_fly.logging_config import get_logger, setup_logging

logger = get_logger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[3]
CONFIG_DIR = str(PROJECT_ROOT / "configs")


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
    val_relative = cfg.data.val_split / (cfg.data.train_test_split + cfg.data.val_split)

    # First carve off the test set, then split the remainder into train/val.
    train_val, test = train_test_split(
        df, test_size=test_split, random_state=cfg.data.random_state
    )
    train, val = train_test_split(
        train_val, test_size=val_relative, random_state=cfg.data.random_state
    )

    train_path = output_dir / "train.csv"
    val_path = output_dir / "val.csv"
    test_path = output_dir / "test.csv"

    train.to_csv(train_path, index=False)
    val.to_csv(val_path, index=False)
    test.to_csv(test_path, index=False)

    logger.info(
        "Wrote splits to %s: train=%d, val=%d, test=%d",
        output_dir,
        len(train),
        len(val),
        len(test),
    )
    logger.info("Data processing complete. Run `make train` next.")


if __name__ == "__main__":
    main()
