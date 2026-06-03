"""Pydantic request/response models for the inference API.

Field names mirror the *raw* dataset columns the trained scikit-learn
pipeline consumes (see ``train_model.py``). The pipeline's
``ColumnTransformer`` uses ``remainder="drop"``, so only the columns
referenced here are required; everything else is ignored.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

# One realistic Zomato-style record, reused as the OpenAPI example.
_EXAMPLE: dict[str, Any] = {
    "Delivery_person_Age": 30.0,
    "Delivery_person_Ratings": 4.5,
    "Restaurant_latitude": 22.745049,
    "Restaurant_longitude": 75.892471,
    "Delivery_location_latitude": 22.765049,
    "Delivery_location_longitude": 75.912471,
    "Order_Date": "19-03-2022",
    "Time_Orderd": "17:30",
    "Weather_conditions": "Sunny",
    "Road_traffic_density": "Low",
    "Vehicle_condition": 2,
    "Type_of_order": "Snack",
    "Type_of_vehicle": "motorcycle",
    "multiple_deliveries": 0.0,
    "Festival": "No",
    "City": "Urban",
}


class DeliveryRequest(BaseModel):
    """A single delivery scenario to estimate the time for.

    Column names intentionally match the training data exactly so the
    payload can be turned into a DataFrame and fed straight to the
    pipeline without renaming.
    """

    Delivery_person_Age: float = Field(..., description="Courier age in years.")
    Delivery_person_Ratings: float = Field(
        ..., description="Courier average rating (0-5)."
    )
    Restaurant_latitude: float
    Restaurant_longitude: float
    Delivery_location_latitude: float
    Delivery_location_longitude: float
    Order_Date: str = Field(..., description="Order date, day-first e.g. '19-03-2022'.")
    Time_Orderd: str = Field(..., description="Order time 'HH:MM' (24h).")
    Weather_conditions: str
    Road_traffic_density: str = Field(
        ..., description="One of Low / Medium / High / Jam."
    )
    Vehicle_condition: int
    Type_of_order: str
    Type_of_vehicle: str
    multiple_deliveries: float = Field(
        ..., description="Number of concurrent deliveries on the trip."
    )
    Festival: str = Field(..., description="'Yes' or 'No'.")
    City: str = Field(..., description="One of Urban / Metropolitian / Semi-Urban.")

    model_config = {"json_schema_extra": {"example": _EXAMPLE}}


class BatchDeliveryRequest(BaseModel):
    """Wrapper for scoring many records in one call."""

    instances: list[DeliveryRequest] = Field(..., min_length=1)

    model_config = {"json_schema_extra": {"example": {"instances": [_EXAMPLE]}}}


class PredictionResponse(BaseModel):
    """Single prediction result."""

    predicted_time_min: float = Field(
        ..., description="Estimated delivery time in minutes."
    )
    model_version: str = Field(..., description="Identifier of the served model.")


class BatchPredictionResponse(BaseModel):
    """Batch prediction result, order-aligned with the request."""

    predictions: list[float]
    model_version: str
    count: int


class HealthResponse(BaseModel):
    """Liveness / readiness signal."""

    status: str
    model_loaded: bool
    model_version: str | None = None
