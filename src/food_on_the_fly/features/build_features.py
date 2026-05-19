"""Feature engineering transformations."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from omegaconf import DictConfig
from sklearn.base import BaseEstimator, TransformerMixin

from food_on_the_fly.logging_config import get_logger

logger = get_logger(__name__)


class HaversineTransformer(BaseEstimator, TransformerMixin):
    def __init__(
        self, lat_col1: str, lon_col1: str, lat_col2: str, lon_col2: str
    ) -> None:
        self.lat_col1 = lat_col1
        self.lon_col1 = lon_col1
        self.lat_col2 = lat_col2
        self.lon_col2 = lon_col2

    def fit(self, X: pd.DataFrame, y: Any = None) -> HaversineTransformer:
        cols = [self.lat_col1, self.lon_col1, self.lat_col2, self.lon_col2]
        missing = [c for c in cols if c not in X.columns]
        if missing:
            raise ValueError(f"HaversineTransformer: missing columns {missing}")
        return self

    def transform(self, X: pd.DataFrame) -> np.ndarray:
        lat1, lon1, lat2, lon2 = map(
            np.radians,
            [X[self.lat_col1], X[self.lon_col1], X[self.lat_col2], X[self.lon_col2]],
        )
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        a = (
            np.sin(dlat / 2.0) ** 2
            + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0) ** 2
        )
        c = 2 * np.arcsin(np.sqrt(a))
        return np.asarray((6371 * c).values).reshape(-1, 1)

    def get_feature_names_out(
        self, input_features: list[str] | None = None
    ) -> np.ndarray:
        return np.array(["distance_km"])


class WeekdayTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, date_col: str, weekday_end: int = 4) -> None:
        self.date_col = date_col
        self.weekday_end = weekday_end

    def fit(self, X: pd.DataFrame, y: Any = None) -> WeekdayTransformer:
        if self.date_col not in X.columns:
            raise ValueError(f"WeekdayTransformer: missing column {self.date_col}")
        return self

    def transform(self, X: pd.DataFrame) -> np.ndarray:
        # Changes the date column to 0-6 where 0 is Monday and 6 is Sunday,
        # then creates a binary feature for weekday vs weekend
        weekdays = pd.to_datetime(X[self.date_col], dayfirst=True).dt.weekday
        return np.asarray((weekdays < self.weekday_end).astype(int).values).reshape(
            -1, 1
        )

    def get_feature_names_out(
        self, input_features: list[str] | None = None
    ) -> np.ndarray:
        return np.array(["is_weekday"])


class RushHourTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, time_col: str) -> None:
        self.time_col = time_col

    def fit(self, X: pd.DataFrame, y: Any = None) -> RushHourTransformer:
        if self.time_col not in X.columns:
            raise ValueError(f"RushHourTransformer: missing column {self.time_col}")
        return self

    def transform(self, X: pd.DataFrame) -> np.ndarray:
        # Creates a binary feature for rush hour (7-9am and 5-7pm)
        def parse_hour(val: Any) -> int:
            try:
                if isinstance(val, str) and ":" in val:
                    return int(val.split(":")[0])
                else:
                    return int(float(val) * 24)
            except (ValueError, AttributeError):
                return 0

        hours = X[self.time_col].apply(parse_hour)
        is_rush_hour = ((hours >= 11) & (hours <= 13)) | ((hours >= 18) & (hours <= 21))
        return np.asarray(is_rush_hour.astype(int).values).reshape(-1, 1)

    def get_feature_names_out(
        self, input_features: list[str] | None = None
    ) -> np.ndarray:
        return np.array(["is_rush_hour"])


def create_haversine_transformer(config: DictConfig) -> HaversineTransformer:
    """Factory function to create a HaversineTransformer from config."""
    params = config.features.haversine
    return HaversineTransformer(
        lat_col1=params.lat_col1,
        lon_col1=params.lon_col1,
        lat_col2=params.lat_col2,
        lon_col2=params.lon_col2,
    )


def create_weekday_transformer(config: DictConfig) -> WeekdayTransformer:
    """Factory function to create a WeekdayTransformer from config."""
    params = config.features.weekday
    return WeekdayTransformer(
        date_col=params.date_col,
        weekday_end=params.weekday_end,
    )


def create_rush_hour_transformer(config: DictConfig) -> RushHourTransformer:
    """Factory function to create a RushHourTransformer from config."""
    params = config.features.rush_hour
    return RushHourTransformer(
        time_col=params.time_col,
    )


def build_features(df: pd.DataFrame, config: DictConfig) -> pd.DataFrame:
    """Derive model-ready features from a processed dataframe.

    Keep transformations deterministic and side-effect free so the
    function can be reused at inference time.
    """
    logger.info("Building features for %d rows", len(df))
    return df.copy()
