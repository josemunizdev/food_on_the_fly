"""


make_dataset.py - Data ingestion and train/val/test splitting
Imran Ahmed

Dataset: saurabhbadole/zomato-delivery-operations-analytics-dataset
"""

import hashlib
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

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
min_date = df["Order_Date"].min()
max_date = df["Order_Date"].max()
print(f"Date range: {min_date.date()} to {max_date.date()}")

# temporal split: first 40 days for modeling, last 14 days for drift monitoring
cutoff = min_date + pd.Timedelta(days=40)
modeling_data = df[df["Order_Date"] <= cutoff]
drift_data = df[df["Order_Date"] > cutoff]
print(f"\nFirst 40 days (modeling): {len(modeling_data):,} rows")
print(f"Last 14 days (drift):    {len(drift_data):,} rows")

# 70/15/15 split on the first 40 days only
train, temp = train_test_split(modeling_data, test_size=0.30, random_state=42)
val, test = train_test_split(temp, test_size=0.50, random_state=42)

print(f"\nTrain: {len(train):,} | Val: {len(val):,} | Test: {len(test):,}")

# save to data/processed/
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
train.to_csv(PROCESSED_DIR / "train.csv", index=False)
val.to_csv(PROCESSED_DIR / "val.csv", index=False)
test.to_csv(PROCESSED_DIR / "test.csv", index=False)
drift_data.to_csv(PROCESSED_DIR / "drift.csv", index=False)

# print md5 hashes for DVC
print("\nMD5 hashes:")
for name in ["train", "val", "test", "drift"]:
    path = PROCESSED_DIR / f"{name}.csv"
    h = hashlib.md5(open(path, "rb").read()).hexdigest()
    print(f"  {name}.csv -> {h}")

print("\nDone")
