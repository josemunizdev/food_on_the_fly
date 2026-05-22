from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from food_on_the_fly.features.build_features import (
    HaversineTransformer,
    RushHourTransformer,
    WeekdayTransformer,
)

EARTH_RADIUS_KM = 6371


class TestHaversineTransformer:
    @pytest.fixture()
    def transformer(self):
        return HaversineTransformer(
            lat_col1="Restaurant_latitude",
            lon_col1="Restaurant_longitude",
            lat_col2="Delivery_location_latitude",
            lon_col2="Delivery_location_longitude",
        )

    @pytest.fixture()
    def sample_df(self):
        return pd.DataFrame(
            {
                "Restaurant_latitude": [28.6139, 19.0760, 12.9716],
                "Restaurant_longitude": [77.2090, 72.8777, 77.5946],
                "Delivery_location_latitude": [28.6300, 19.0900, 12.9800],
                "Delivery_location_longitude": [77.2200, 72.8900, 77.6000],
            }
        )

    def test_known_distance_delhi_to_mumbai(self, transformer):
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

    def test_same_point_returns_zero(self, transformer):
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

    def test_output_shape(self, transformer, sample_df):
        transformer.fit(sample_df)
        result = transformer.transform(sample_df)
        assert result.shape == (len(sample_df), 1)

    def test_no_negative_distances(self, transformer, sample_df):
        transformer.fit(sample_df)
        result = transformer.transform(sample_df)
        assert np.all(result >= 0)

    def test_symmetry(self):
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

    def test_missing_column_raises(self, transformer):
        bad_df = pd.DataFrame({"Restaurant_latitude": [1.0]})
        with pytest.raises(ValueError, match="missing columns"):
            transformer.fit(bad_df)

    def test_feature_names_out(self, transformer):
        names = transformer.get_feature_names_out()
        assert list(names) == ["distance_km"]

    def test_short_delivery_distance(self, transformer):
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
    def transformer(self):
        return WeekdayTransformer(date_col="Order_Date")

    def test_monday_is_weekday(self, transformer):
        df = pd.DataFrame({"Order_Date": ["13-01-2025"]})
        transformer.fit(df)
        result = transformer.transform(df)
        assert result[0, 0] == 1

    def test_saturday_is_weekend(self, transformer):
        df = pd.DataFrame({"Order_Date": ["18-01-2025"]})
        transformer.fit(df)
        result = transformer.transform(df)
        assert result[0, 0] == 0

    def test_sunday_is_weekend(self, transformer):
        df = pd.DataFrame({"Order_Date": ["19-01-2025"]})
        transformer.fit(df)
        result = transformer.transform(df)
        assert result[0, 0] == 0

    def test_output_shape(self, transformer):
        df = pd.DataFrame({"Order_Date": ["13-01-2025", "14-01-2025", "18-01-2025"]})
        transformer.fit(df)
        result = transformer.transform(df)
        assert result.shape == (3, 1)

    def test_output_is_binary(self, transformer):
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
        assert set(result.flatten()).issubset({0, 1})

    def test_missing_column_raises(self, transformer):
        bad_df = pd.DataFrame({"wrong_col": ["13-01-2025"]})
        with pytest.raises(ValueError, match="missing column"):
            transformer.fit(bad_df)

    def test_feature_names_out(self, transformer):
        names = transformer.get_feature_names_out()
        assert list(names) == ["is_weekday"]


class TestRushHourTransformer:
    @pytest.fixture()
    def transformer(self):
        return RushHourTransformer(time_col="Time_Orderd")

    def test_lunch_rush_is_rush(self, transformer):
        df = pd.DataFrame({"Time_Orderd": ["12:30"]})
        transformer.fit(df)
        result = transformer.transform(df)
        assert result[0, 0] == 1

    def test_dinner_rush_is_rush(self, transformer):
        df = pd.DataFrame({"Time_Orderd": ["19:00"]})
        transformer.fit(df)
        result = transformer.transform(df)
        assert result[0, 0] == 1

    def test_early_morning_not_rush(self, transformer):
        df = pd.DataFrame({"Time_Orderd": ["6:00"]})
        transformer.fit(df)
        result = transformer.transform(df)
        assert result[0, 0] == 0

    def test_mid_afternoon_not_rush(self, transformer):
        df = pd.DataFrame({"Time_Orderd": ["15:00"]})
        transformer.fit(df)
        result = transformer.transform(df)
        assert result[0, 0] == 0

    def test_output_shape(self, transformer):
        df = pd.DataFrame({"Time_Orderd": ["8:00", "12:00", "15:00", "19:00"]})
        transformer.fit(df)
        result = transformer.transform(df)
        assert result.shape == (4, 1)

    def test_output_is_binary(self, transformer):
        df = pd.DataFrame({"Time_Orderd": ["6:00", "12:00", "15:00", "19:00", "23:00"]})
        transformer.fit(df)
        result = transformer.transform(df)
        assert set(result.flatten()).issubset({0, 1})

    def test_missing_column_raises(self, transformer):
        bad_df = pd.DataFrame({"wrong_col": ["12:00"]})
        with pytest.raises(ValueError, match="missing column"):
            transformer.fit(bad_df)

    def test_feature_names_out(self, transformer):
        names = transformer.get_feature_names_out()
        assert list(names) == ["is_rush_hour"]
