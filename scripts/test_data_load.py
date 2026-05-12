"""Smoke test for GCS data read.

Usage (from repo root, with .venv activated):
    python scripts/test_data_load.py

Confirms that pandas + gcsfs can read the dataset from GCS using the
caller's Application Default Credentials. If this works, training
pipelines that read from GCS will work too.
"""

from __future__ import annotations

from food_on_the_fly.data.loaders import load_dataset

GCS_URI = "gs://project-9aed1f8e-f40e-4a15-858-models/data/raw/zomato_dataset.csv"


def main() -> None:
    df = load_dataset(GCS_URI)
    print("Shape:", df.shape)
    print("Columns:", list(df.columns))
    print()
    print(df.head())


if __name__ == "__main__":
    main()
