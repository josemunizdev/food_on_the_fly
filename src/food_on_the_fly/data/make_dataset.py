"""

make_dataset.py - Data ingestion and train/val/test splitting
Imran Ahmed
Dataset: saurabhbadole/zomato-delivery-operations-analytics-dataset
Run from repo root: python src/food_on_the_fly/data/make_dataset.py
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import hashlib
import os
from pathlib import Path

# resolve paths relative to repo structure
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[2]
RAW_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

# load raw data
RAW_PATH = RAW_DIR / "Zomato Dataset.csv"

if not RAW_PATH.exists():
    print(f"ERROR: Raw dataset not found at {RAW_PATH}")
    print(f"Copy 'Zomato Dataset.csv' from your Downloads into: {RAW_DIR}/")
    exit(1)

df = pd.read_csv(RAW_PATH)
print(f"Loaded {df.shape[0]:,} rows, {df.shape[1]} columns")

# clean target column
df["Time_taken (min)"] = pd.to_numeric(
    df["Time_taken (min)"].astype(str).str.strip(), errors="coerce"
)

# filter out zero/negative times and missing targets
before = len(df)
df = df[df["Time_taken (min)"] > 0]
df = df.dropna(subset=["Time_taken (min)"])
print(f"Filtered {before - len(df)} bad rows, {len(df):,} remaining")

# parse dates
df["Order_Date"] = pd.to_datetime(df["Order_Date"], format="%d-%m-%Y", errors="coerce")
print(f"Date range: {df['Order_Date'].min().date()} to {df['Order_Date'].max().date()}")

# 70/15/15 split
train, temp = train_test_split(df, test_size=0.30, random_state=42)
val, test = train_test_split(temp, test_size=0.50, random_state=42)

print(f"Train: {len(train):,} | Val: {len(val):,} | Test: {len(test):,}")

# save to data/processed/
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
train.to_csv(PROCESSED_DIR / "train.csv", index=False)
val.to_csv(PROCESSED_DIR / "val.csv", index=False)
test.to_csv(PROCESSED_DIR / "test.csv", index=False)

# print md5 hashes for DVC
print("\nMD5 hashes:")
for name in ["train", "val", "test"]:
    path = PROCESSED_DIR / f"{name}.csv"
    h = hashlib.md5(open(path, "rb").read()).hexdigest()
    print(f"  {name}.csv -> {h}")

print(f"\nDone")