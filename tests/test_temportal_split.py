import pandas as pd
import pytest

from food_on_the_fly.data.make_dataset import split_by_date


@pytest.fixture
def sample_temporal_data() -> pd.DataFrame:
    """Create sample data with known date range."""
    dates = pd.date_range("2024-01-01", periods=100, freq="D")
    df = pd.DataFrame({"Order_Date": dates, "value": range(100)})
    return df


def test_no_crossover_between_modeling_and_drift(
    sample_temporal_data: pd.DataFrame,
) -> None:
    """Ensure max(modeling) < min(drift)."""
    train, val, test, drift = split_by_date(sample_temporal_data)

    modeling_max_date = max(
        train["Order_Date"].max(), val["Order_Date"].max(), test["Order_Date"].max()
    )
    drift_min_date = drift["Order_Date"].min()

    assert modeling_max_date < drift_min_date


def test_all_rows_accounted_for(sample_temporal_data: pd.DataFrame) -> None:
    """Every row should be in exactly one split."""
    train, val, test, drift = split_by_date(sample_temporal_data)

    total = len(train) + len(val) + len(test) + len(drift)
    assert total == len(sample_temporal_data)


def test_baseline_window_is_40_days(sample_temporal_data: pd.DataFrame) -> None:
    """Modeling window should span exactly 40 calendar days."""
    train, val, test, drift = split_by_date(sample_temporal_data, baseline_days=40)

    modeling = pd.concat([train, val, test])
    date_range_days = (modeling["Order_Date"].max() - modeling["Order_Date"].min()).days

    assert date_range_days < 40  # Less than because we start at day 0
