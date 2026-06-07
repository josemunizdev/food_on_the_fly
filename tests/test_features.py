from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from food_on_the_fly.features.build_features import (
    DayOfWeekTransformer,
    HaversineTransformer,
    HourOfDayTransformer,
    RushHourTransformer,
    WeekdayTransformer,
)

EARTH_RADIUS_KM = 6371


class TestHaversineTransformer:
    @pytest.fixture()
    def transformer(self) -> HaversineTransformer:
        return HaversineTransformer(
            lat_col1="Restaurant_latitude",
            lon_col1="Restaurant_longitude",
            lat_col2="Delivery_location_latitude",
            lon_col2="Delivery_location_longitude",
        )

    @pytest.fixture()
    def sample_df(self) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "Restaurant_latitude": [28.6139, 19.0760, 12.9716],
                "Restaurant_longitude": [77.2090, 72.8777, 77.5946],
                "Delivery_location_latitude": [28.6300, 19.0900, 12.9800],
                "Delivery_location_longitude": [77.2200, 72.8900, 77.6000],
            }
        )

    def test_known_distance_delhi_to_mumbai(
        self, transformer: HaversineTransformer
    ) -> None:
        df = pd.DataFrame(
            {
                "Restaurant_latitude": [28.6139],
                "Restaurant_longitude": [77.2090],
                "Delivery_location_latitude": [19.0760],
                "Delivery_location_longitude": [72.8777],
            }
        )
        transformer.fit(df)
        result = transformer.transform(df)
        assert result.shape == (1, 1)
        assert 1140 < result[0, 0] < 1160

    def test_same_point_returns_zero(self, transformer: HaversineTransformer) -> None:
        df = pd.DataFrame(
            {
                "Restaurant_latitude": [28.6139],
                "Restaurant_longitude": [77.2090],
                "Delivery_location_latitude": [28.6139],
                "Delivery_location_longitude": [77.2090],
            }
        )
        transformer.fit(df)
        result = transformer.transform(df)
        assert result[0, 0] == pytest.approx(0.0, abs=1e-6)

    def test_output_shape(
        self, transformer: HaversineTransformer, sample_df: pd.DataFrame
    ) -> None:
        transformer.fit(sample_df)
        result = transformer.transform(sample_df)
        assert result.shape == (len(sample_df), 1)

    def test_no_negative_distances(
        self, transformer: HaversineTransformer, sample_df: pd.DataFrame
    ) -> None:
        transformer.fit(sample_df)
        result = transformer.transform(sample_df)
        assert np.all(result >= 0)

    def test_symmetry(self) -> None:
        t_forward = HaversineTransformer("a_lat", "a_lon", "b_lat", "b_lon")
        t_reverse = HaversineTransformer("b_lat", "b_lon", "a_lat", "a_lon")
        df = pd.DataFrame(
            {
                "a_lat": [28.6139],
                "a_lon": [77.2090],
                "b_lat": [19.0760],
                "b_lon": [72.8777],
            }
        )
        t_forward.fit(df)
        t_reverse.fit(df)
        d1 = t_forward.transform(df)[0, 0]
        d2 = t_reverse.transform(df)[0, 0]
        assert d1 == pytest.approx(d2, rel=1e-6)

    def test_missing_column_raises(self, transformer: HaversineTransformer) -> None:
        bad_df = pd.DataFrame({"Restaurant_latitude": [1.0]})
        with pytest.raises(ValueError, match="missing columns"):
            transformer.fit(bad_df)

    def test_feature_names_out(self, transformer: HaversineTransformer) -> None:
        names = transformer.get_feature_names_out()
        assert list(names) == ["distance_km"]

    def test_short_delivery_distance(self, transformer: HaversineTransformer) -> None:
        df = pd.DataFrame(
            {
                "Restaurant_latitude": [12.9716],
                "Restaurant_longitude": [77.5946],
                "Delivery_location_latitude": [12.9750],
                "Delivery_location_longitude": [77.5980],
            }
        )
        transformer.fit(df)
        result = transformer.transform(df)
        assert 0 < result[0, 0] < 5


class TestWeekdayTransformer:
    @pytest.fixture()
    def transformer(self) -> WeekdayTransformer:
        return WeekdayTransformer(date_col="Order_Date")

    def test_monday_is_weekday(self, transformer: WeekdayTransformer) -> None:
        df = pd.DataFrame({"Order_Date": ["13-01-2025"]})
        transformer.fit(df)
        result = transformer.transform(df)
        assert result["is_weekday"].iloc[0] == 1

    def test_saturday_is_weekend(self, transformer: WeekdayTransformer) -> None:
        df = pd.DataFrame({"Order_Date": ["18-01-2025"]})
        transformer.fit(df)
        result = transformer.transform(df)
        assert result["is_weekday"].iloc[0] == 0

    def test_sunday_is_weekend(self, transformer: WeekdayTransformer) -> None:
        df = pd.DataFrame({"Order_Date": ["19-01-2025"]})
        transformer.fit(df)
        result = transformer.transform(df)
        assert result["is_weekday"].iloc[0] == 0

    def test_output_shape(self, transformer: WeekdayTransformer) -> None:
        df = pd.DataFrame({"Order_Date": ["13-01-2025", "14-01-2025", "18-01-2025"]})
        transformer.fit(df)
        result = transformer.transform(df)
        assert result.shape == (3, 1)

    def test_output_is_binary(self, transformer: WeekdayTransformer) -> None:
        df = pd.DataFrame(
            {
                "Order_Date": [
                    "13-01-2025",
                    "14-01-2025",
                    "15-01-2025",
                    "18-01-2025",
                    "19-01-2025",
                ]
            }
        )
        transformer.fit(df)
        result = transformer.transform(df)
        assert set(result["is_weekday"].values).issubset({0, 1})

    def test_missing_column_raises(self, transformer: WeekdayTransformer) -> None:
        bad_df = pd.DataFrame({"wrong_col": ["13-01-2025"]})
        with pytest.raises(ValueError, match="missing column"):
            transformer.fit(bad_df)

    def test_feature_names_out(self, transformer: WeekdayTransformer) -> None:
        names = transformer.get_feature_names_out()
        assert list(names) == ["is_weekday"]


class TestRushHourTransformer:
    @pytest.fixture()
    def transformer(self) -> RushHourTransformer:
        return RushHourTransformer(time_col="Time_Orderd")

    def test_lunch_rush_is_rush(self, transformer: RushHourTransformer) -> None:
        df = pd.DataFrame({"Time_Orderd": ["12:30"]})
        transformer.fit(df)
        result = transformer.transform(df)
        assert result[0, 0] == 1

    def test_dinner_rush_is_rush(self, transformer: RushHourTransformer) -> None:
        df = pd.DataFrame({"Time_Orderd": ["19:00"]})
        transformer.fit(df)
        result = transformer.transform(df)
        assert result[0, 0] == 1

    def test_early_morning_not_rush(self, transformer: RushHourTransformer) -> None:
        df = pd.DataFrame({"Time_Orderd": ["6:00"]})
        transformer.fit(df)
        result = transformer.transform(df)
        assert result[0, 0] == 0

    def test_mid_afternoon_not_rush(self, transformer: RushHourTransformer) -> None:
        df = pd.DataFrame({"Time_Orderd": ["15:00"]})
        transformer.fit(df)
        result = transformer.transform(df)
        assert result[0, 0] == 0

    def test_output_shape(self, transformer: RushHourTransformer) -> None:
        df = pd.DataFrame({"Time_Orderd": ["8:00", "12:00", "15:00", "19:00"]})
        transformer.fit(df)
        result = transformer.transform(df)
        assert result.shape == (4, 1)

    def test_output_is_binary(self, transformer: RushHourTransformer) -> None:
        df = pd.DataFrame({"Time_Orderd": ["6:00", "12:00", "15:00", "19:00", "23:00"]})
        transformer.fit(df)
        result = transformer.transform(df)
        assert set(result.flatten()).issubset({0, 1})

    def test_missing_column_raises(self, transformer: RushHourTransformer) -> None:
        bad_df = pd.DataFrame({"wrong_col": ["12:00"]})
        with pytest.raises(ValueError, match="missing column"):
            transformer.fit(bad_df)

    def test_feature_names_out(self, transformer: RushHourTransformer) -> None:
        names = transformer.get_feature_names_out()
        assert list(names) == ["is_rush_hour"]


class TestDayOfWeekTransformer:
    """Tests for DayOfWeekTransformer."""

    def test_monday_is_zero(self) -> None:
        """Monday should be encoded as 0."""
        df = pd.DataFrame({"Order_Date": ["01-01-2024"]})  # This is a Monday
        transformer = DayOfWeekTransformer()

        result = transformer.fit_transform(df)

        assert result["day_of_week"].iloc[0] == 0

    def test_sunday_is_six(self) -> None:
        """Sunday should be encoded as 6."""
        df = pd.DataFrame({"Order_Date": ["07-01-2024"]})  # This is a Sunday
        transformer = DayOfWeekTransformer()

        result = transformer.fit_transform(df)

        assert result["day_of_week"].iloc[0] == 6


class TestHourOfDayTransformer:
    """Tests for HourOfDayTransformer."""

    def test_extracts_hour_correctly(self) -> None:
        """Should extract hour from time string."""
        df = pd.DataFrame({"Time_Orderd": ["12:30:45", "18:15:30", "06:00:00"]})
        transformer = HourOfDayTransformer()

        result = transformer.fit_transform(df)

        assert result["hour_of_day"].tolist() == [12, 18, 6]

    def test_output_shape(self) -> None:
        """Should return DataFrame with one column."""
        df = pd.DataFrame({"Time_Orderd": ["12:00:00"] * 5})
        transformer = HourOfDayTransformer()

        result = transformer.fit_transform(df)

        assert result.shape == (5, 1)

    def test_feature_names_out(self) -> None:
        """Should return correct feature name."""
        transformer = HourOfDayTransformer()

        names = transformer.get_feature_names_out()

        assert names[0] == "hour_of_day"
