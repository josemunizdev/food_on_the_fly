"""Tests for data loading functions."""

import pandas as pd
import pytest

from food_on_the_fly.data.loaders import (
    load_processed,
    save_processed,
)


def test_load_processed_returns_dataframe() -> None:
    """Test that load_processed returns a DataFrame."""
    # This assumes you have processed data files
    try:
        df = load_processed("train.csv")
        assert isinstance(df, pd.DataFrame)
        assert len(df) > 0
    except FileNotFoundError:
        pytest.skip("Processed data not available")


def test_load_processed_has_expected_columns() -> None:
    """Test that loaded data has expected columns."""
    try:
        df = load_processed("train.csv")
        expected_cols = [
            "Restaurant_latitude",
            "Restaurant_longitude",
            "Delivery_location_latitude",
            "Delivery_location_longitude",
        ]
        for col in expected_cols:
            assert col in df.columns, f"Missing column: {col}"
    except FileNotFoundError:
        pytest.skip("Processed data not available")


def test_save_and_load_processed_roundtrip(tmp_path: pytest.TempPathFactory) -> None:
    """Test saving and loading processed data."""
    # Create sample data
    df = pd.DataFrame(
        {
            "Restaurant_latitude": [12.97, 13.08],
            "Restaurant_longitude": [77.59, 77.61],
            "Delivery_location_latitude": [12.98, 13.09],
            "Delivery_location_longitude": [77.60, 77.62],
            "Time_taken (min)": [20, 25],
        }
    )

    # Save
    filepath = tmp_path / "test.csv"
    save_processed(df, filepath)

    # Load
    loaded_df = pd.read_csv(filepath)

    # Verify
    assert len(loaded_df) == len(df)
    assert list(loaded_df.columns) == list(df.columns)
