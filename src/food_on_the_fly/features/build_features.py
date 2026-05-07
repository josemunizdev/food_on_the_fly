"""Feature engineering transformations."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
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
        return (6371 * c).values.reshape(-1, 1)

    def get_feature_names_out(
        self, input_features: list[str] | None = None
    ) -> np.ndarray:
        return np.array(["distance_km"])


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Derive model-ready features from a processed dataframe.

    Keep transformations deterministic and side-effect free so the
    function can be reused at inference time.
    """
    logger.info("Building features for %d rows", len(df))
    return df.copy()
