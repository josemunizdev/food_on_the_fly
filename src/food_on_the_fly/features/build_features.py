"""Feature engineering transformations."""

from __future__ import annotations

import pandas as pd
import numpy as np

from food_on_the_fly.logging_config import get_logger

logger = get_logger(__name__)



from sklearn.base import BaseEstimator, TransformerMixin

class HaversineTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, lat_col1, lon_col1, lat_col2, lon_col2):
        self.lat_col1 = lat_col1
        self.lon_col1 = lon_col1
        self.lat_col2 = lat_col2
        self.lon_col2 = lon_col2

    def fit(self, X, y=None):
        missing = [c for c in [self.lat_col1, self.lon_col1, self.lat_col2, self.lon_col2] if c not in X.columns]
        if missing:
            raise ValueError(f"HaversineTransformer: missing columns {missing}")
        return self

    def transform(self, X):
        lat1, lon1, lat2, lon2 = map(np.radians, [
            X[self.lat_col1], X[self.lon_col1],
            X[self.lat_col2], X[self.lon_col2]
        ])
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0) ** 2
        c = 2 * np.arcsin(np.sqrt(a))
        return (6371 * c).values.reshape(-1, 1)

    def get_feature_names_out(self, input_features=None):
        return np.array(["distance_km"])


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Derive model-ready features from a processed dataframe.

    Keep transformations deterministic and side-effect free so the
    function can be reused at inference time.
    """
    logger.info("Building features for %d rows", len(df))
    return df.copy()



"""
Column info:
  ID (object): 0xcdcd
  Delivery_person_ID (object): DEHRES17DEL01
  Delivery_person_Age (float64): 36.0
  Delivery_person_Ratings (float64): 4.2
  Restaurant_latitude (float64): 30.327968
  Restaurant_longitude (float64): 78.046106
  Delivery_location_latitude (float64): 30.397968
  Delivery_location_longitude (float64): 78.116106
  Order_Date (object): 12-02-2022
  Time_Orderd (object): 21:55
  Time_Order_picked (object): 22:10
  Weather_conditions (object): Fog | values: ['Cloudy', 'Fog', 'Sandstorms', 'Stormy', 'Sunny', 'Windy']
  Road_traffic_density (object): Jam | values: ['High', 'Jam', 'Low', 'Medium']
  Vehicle_condition (int64): 2 | values: [np.int64(0), np.int64(1), np.int64(2), np.int64(3)]
  Type_of_order (object): Snack | values: ['Buffet', 'Drinks', 'Meal', 'Snack']
  Type_of_vehicle (object): motorcycle | values: ['bicycle', 'electric_scooter', 'motorcycle', 'scooter']
  multiple_deliveries (float64): 3.0
  Festival (object): No
  City (object): Metropolitian | values: ['Metropolitian', 'Semi-Urban', 'Urban']
  Time_taken (min) (int64): 46
"""