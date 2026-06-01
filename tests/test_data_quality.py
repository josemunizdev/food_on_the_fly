from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

try:
    from food_on_the_fly.config import PROJECT_ROOT
except ImportError:
    PROJECT_ROOT = Path(__file__).resolve().parents[1]

PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
RAW_DIR = PROJECT_ROOT / "data" / "raw"

INDIA_BOUNDS = {
    "lat_min": 6.0,
    "lat_max": 37.0,
    "lon_min": 67.0,
    "lon_max": 98.0,
}

COORD_COLS = [
    "Restaurant_latitude",
    "Restaurant_longitude",
    "Delivery_location_latitude",
    "Delivery_location_longitude",
]


def _load_data() -> pd.DataFrame:
    train_path = PROCESSED_DIR / "train.csv"
    raw_path = RAW_DIR / "Zomato Dataset.csv"

    if train_path.exists():
        return pd.read_csv(train_path)
    elif raw_path.exists():
        return pd.read_csv(raw_path)
    else:
        pytest.skip("No dataset found")


@pytest.fixture(scope="module")
def df() -> pd.DataFrame:
    return _load_data()


class TestNoNegativeDeliveryTimes:
    def test_no_negative_values(self, df: pd.DataFrame) -> None:
        col = "Time_taken (min)"
        times = pd.to_numeric(df[col], errors="coerce")
        negatives = df[times < 0]
        assert len(negatives) == 0, (
            f"{len(negatives)} rows have negative delivery times"
        )

    def test_no_zero_values(self, df: pd.DataFrame) -> None:
        col = "Time_taken (min)"
        times = pd.to_numeric(df[col], errors="coerce")
        zeros = df[times == 0]
        assert len(zeros) == 0, f"{len(zeros)} rows have zero delivery time"

    def test_no_null_values(self, df: pd.DataFrame) -> None:
        col = "Time_taken (min)"
        nulls = pd.to_numeric(df[col], errors="coerce").isna().sum()
        assert nulls == 0, f"{nulls} rows have null delivery times"

    def test_within_reasonable_range(self, df: pd.DataFrame) -> None:
        col = "Time_taken (min)"
        times = pd.to_numeric(df[col], errors="coerce").dropna()
        out = times[(times < 1) | (times > 180)]
        assert len(out) == 0, f"{len(out)} rows outside [1, 180] min"


class TestLatLonBoundsMatchCityCoordinates:
    def _nonzero(self, series: pd.Series) -> pd.Series:
        vals = pd.to_numeric(series, errors="coerce").dropna()
        return vals[vals != 0.0]

    def test_restaurant_latitude_in_bounds(self, df: pd.DataFrame) -> None:
        lats = self._nonzero(df["Restaurant_latitude"])
        bad = lats[(lats < INDIA_BOUNDS["lat_min"]) | (lats > INDIA_BOUNDS["lat_max"])]
        pct = len(bad) / len(lats) * 100
        assert pct < 2, f"{len(bad)} restaurant lats ({pct:.1f}%) outside India bounds"

    def test_restaurant_longitude_in_bounds(self, df: pd.DataFrame) -> None:
        lons = self._nonzero(df["Restaurant_longitude"])
        bad = lons[(lons < INDIA_BOUNDS["lon_min"]) | (lons > INDIA_BOUNDS["lon_max"])]
        pct = len(bad) / len(lons) * 100
        assert pct < 2, f"{len(bad)} restaurant lons ({pct:.1f}%) outside India bounds"

    def test_delivery_latitude_in_bounds(self, df: pd.DataFrame) -> None:
        lats = self._nonzero(df["Delivery_location_latitude"])
        bad = lats[(lats < INDIA_BOUNDS["lat_min"]) | (lats > INDIA_BOUNDS["lat_max"])]
        pct = len(bad) / len(lats) * 100
        assert pct < 15, f"{len(bad)} delivery lats ({pct:.1f}%) outside India bounds"

    def test_delivery_longitude_in_bounds(self, df: pd.DataFrame) -> None:
        lons = self._nonzero(df["Delivery_location_longitude"])
        bad = lons[(lons < INDIA_BOUNDS["lon_min"]) | (lons > INDIA_BOUNDS["lon_max"])]
        pct = len(bad) / len(lons) * 100
        assert pct < 15, f"{len(bad)} delivery lons ({pct:.1f}%) outside India bounds"

    def test_zero_coordinate_percentage(self, df: pd.DataFrame) -> None:
        for col in COORD_COLS:
            if col in df.columns:
                zeros = (df[col] == 0.0).sum()
                pct = zeros / len(df) * 100
                assert pct < 15, (
                    f"{col}: {zeros} rows ({pct:.1f}%) are zero placeholders"
                )

    def test_no_null_coordinates(self, df: pd.DataFrame) -> None:
        for col in COORD_COLS:
            if col in df.columns:
                nulls = df[col].isna().sum()
                assert nulls == 0, f"{col} has {nulls} null values"

    def test_coordinates_globally_valid(self, df: pd.DataFrame) -> None:
        checks = {
            "Restaurant_latitude": (-90, 90),
            "Restaurant_longitude": (-180, 180),
        }
        for col, (lo, hi) in checks.items():
            if col in df.columns:
                vals = pd.to_numeric(df[col], errors="coerce").dropna()
                bad = vals[(vals < lo) | (vals > hi)]
                assert len(bad) == 0, f"{col}: {len(bad)} values outside [{lo}, {hi}]"

    def test_restaurant_delivery_not_same_location(self, df: pd.DataFrame) -> None:
        non_zero = df[
            (df["Restaurant_latitude"] != 0.0) & (df["Restaurant_longitude"] != 0.0)
        ]
        same = non_zero[
            (non_zero["Delivery_location_latitude"] == 0.0)
            & (non_zero["Delivery_location_longitude"] == 0.0)
        ]
        pct = len(same) / len(non_zero) * 100
        assert pct < 5, f"{len(same)} rows ({pct:.1f}%) have zero delivery offset"

    def test_city_column_values(self, df: pd.DataFrame) -> None:
        if "City" not in df.columns:
            pytest.skip("City column not found")
        expected = {"Metropolitian", "Urban", "Semi-Urban"}
        actual = set(df["City"].dropna().unique())
        unexpected = actual - expected
        assert unexpected == set(), f"Unexpected City values: {unexpected}"


def test_processed_train_file_exists() -> None:
    """Test that processed train file exists."""
    train_path = PROCESSED_DIR / "train.csv"
    if not train_path.exists():
        pytest.skip("Processed data not available")
    assert train_path.exists(), "train.csv not found in processed data"


def test_processed_data_has_target() -> None:
    """Test that processed data has the target column."""
    try:
        train_df = pd.read_csv(PROCESSED_DIR / "train.csv")
        assert "Time_taken (min)" in train_df.columns
    except FileNotFoundError:
        pytest.skip("Processed data not available")


def test_no_missing_values_in_critical_columns() -> None:
    """Test that critical columns have no missing values."""
    try:
        train_df = pd.read_csv(PROCESSED_DIR / "train.csv")
        critical_cols = COORD_COLS + ["Time_taken (min)"]
        for col in critical_cols:
            if col in train_df.columns:
                assert train_df[col].notna().all(), f"{col} has missing values"
    except FileNotFoundError:
        pytest.skip("Processed data not available")
