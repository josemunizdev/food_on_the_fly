"""Dataset loading utilities.

Wrappers around pandas/numpy I/O that resolve paths against the
project's configured data directories. Supports both local paths
and `gs://` URIs (via gcsfs) so the same code path works on a
laptop and inside Vertex AI / Cloud Run.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from food_on_the_fly.config import PROCESSED_DATA_DIR, RAW_DATA_DIR
from food_on_the_fly.logging_config import get_logger

logger = get_logger(__name__)


def load_dataset(uri: str) -> pd.DataFrame:
    """Load a CSV from any URI pandas understands (local path or gs://...).

    For GCS URIs, requires `gcsfs` (declared in requirements.txt) and
    Application Default Credentials. Authenticate locally with:

        gcloud auth application-default login

    In GCP runtimes (Vertex AI, Cloud Run, GitHub Actions via WIF) the
    runtime service account is used automatically, no extra setup.
    """
    logger.info("Loading dataset from: %s", uri)
    return pd.read_csv(uri)


def load_raw(filename: str) -> pd.DataFrame:
    """Load a CSV from the local `data/raw` directory.

    Useful for offline / cached work. For the canonical dataset prefer
    `load_dataset(cfg.data.source_uri)` so cloud and local agree.
    """
    path = RAW_DATA_DIR / filename
    logger.info("Loading raw data: %s", path)
    return pd.read_csv(path)


def load_processed(filename: str) -> pd.DataFrame:
    """Load a CSV from the `data/processed` directory."""
    path = PROCESSED_DATA_DIR / filename
    logger.info("Loading processed data: %s", path)
    return pd.read_csv(path)


def save_processed(df: pd.DataFrame, filename: str) -> Path:
    """Write a dataframe to `data/processed`, creating the directory if needed."""
    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
    path = PROCESSED_DATA_DIR / filename
    df.to_csv(path, index=False)
    logger.info("Saved processed data: %s", path)
    return path
