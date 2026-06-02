# create script to see what is in each data file,
# specifically get the oldest and newest instances in each file
"""
 ls -lh data/processed/
total 16304
-rw-r--r--@ 1 avivkatz  staff    92B May 21 12:54 drift.csv.dvc
-rw-r--r--@ 1 avivkatz  staff   629K May 28 11:47 test.csv
-rw-r--r--@ 1 avivkatz  staff    90B May 21 12:54 test.csv.dvc
-rw-r--r--@ 1 avivkatz  staff   2.8M May 28 11:47 train.csv
-rw-r--r--@ 1 avivkatz  staff    92B May 21 12:54 train.csv.dvc
-rw-r--r--@ 1 avivkatz  staff   627K May 28 11:47 val.csv
-rw-r--r--@ 1 avivkatz  staff    89B May 21 12:54 val.csv.dvc
-rw-r--r--@ 1 avivkatz  staff   1.0M May 28 11:47 week1.csv
-rw-r--r--@ 1 avivkatz  staff   1.0M May 28 11:47 week2.csv
-rw-r--r--@ 1 avivkatz  staff   136K May 28 11:47 week3.csv
-rw-r--r--@ 1 avivkatz  staff   369K May 27 14:12 week4.csv
-rw-r--r--@ 1 avivkatz  staff   363K May 27 14:12 week5.csv
-rw-r--r--@ 1 avivkatz  staff   317K May 27 14:12 week6.csv
-rw-r--r--@ 1 avivkatz  staff   354K May 27 14:12 week7.csv
-rw-r--r--@ 1 avivkatz  staff   312K May 27 14:12 week8.csv
"""

from pathlib import Path

import pandas as pd
import pytest

try:
    from food_on_the_fly.config import PROJECT_ROOT
except ImportError:
    PROJECT_ROOT = Path(__file__).resolve().parents[1]

PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
RAW_DIR = PROJECT_ROOT / "data" / "raw"


def _load_data(file_name: str) -> pd.DataFrame:
    processed_path = PROCESSED_DIR / file_name
    raw_path = RAW_DIR / file_name

    if processed_path.exists():
        return pd.read_csv(processed_path)
    elif raw_path.exists():
        return pd.read_csv(raw_path)
    else:
        pytest.skip(f"No dataset found for {file_name}")


for file_name in ["train.csv", "val.csv", "test.csv"]:
    df = _load_data(file_name)
    print(f"File: {file_name}")
    print(f"Number of rows: {len(df)}")
    print(f"Oldest instance: {df['Order_Date'].min()}")
    print(f"Newest instance: {df['Order_Date'].max()}")
    print("-" * 40)

for file_name in [f"week{i}.csv" for i in range(1, 9)]:
    df = _load_data(file_name)
    print(f"File: {file_name}")
    print(f"Number of rows: {len(df)}")
    print(f"Oldest instance: {df['Order_Date'].min()}")
    print(f"Newest instance: {df['Order_Date'].max()}")
    print("-" * 40)


df = pd.read_csv("data/processed/train.csv")
df["Order_Date"] = pd.to_datetime(df["Order_Date"], format="%d-%m-%Y")
print("Train data:")
print(f"Min date: {df['Order_Date'].min()}")
print(f"  Max date: {df['Order_Date'].max()}")
print(f"  Date range: {(df['Order_Date'].max() - df['Order_Date'].min()).days} days")
print(f"  Records: {len(df)}")
print()
print("Date distribution:")
print(df["Order_Date"].value_counts().sort_index().head(10))
