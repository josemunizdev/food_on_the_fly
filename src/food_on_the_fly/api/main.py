"""FastAPI application for Food on the Fly delivery-time inference.

Run locally:
    uvicorn food_on_the_fly.api.main:app --reload --port 8080

The app wraps the trained scikit-learn + XGBoost pipeline. Model
resolution is handled in ``model_loader`` (MLflow URI or local path).
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException

from food_on_the_fly.api import model_loader
from food_on_the_fly.api.schemas import (
    BatchDeliveryRequest,
    BatchPredictionResponse,
    DeliveryRequest,
    HealthResponse,
    PredictionResponse,
)
from food_on_the_fly.logging_config import get_logger, setup_logging

logger = get_logger(__name__)


@asynccontextmanager
async def lifespan(_: FastAPI) -> AsyncIterator[None]:
    """Configure logging and warm the model on startup."""
    setup_logging()
    if model_loader.try_load():
        logger.info("Model ready: %s", model_loader.model_version())
    else:
        logger.warning("Starting without a model; /predict will return 503.")
    yield


app = FastAPI(
    title="Food on the Fly - Delivery Time API",
    description="Estimate food-delivery time (minutes) from order details.",
    version="1.0.0",
    lifespan=lifespan,
)


@app.get("/", tags=["meta"])
def root() -> dict[str, str]:
    """Service banner with links to docs."""
    return {
        "service": "food-on-the-fly-api",
        "docs": "/docs",
        "health": "/health",
    }


@app.get("/health", response_model=HealthResponse, tags=["meta"])
def health() -> HealthResponse:
    """Liveness probe. Always 200 so the platform keeps the instance up;
    ``model_loaded`` distinguishes ready from degraded."""
    return HealthResponse(
        status="ok",
        model_loaded=model_loader.is_loaded(),
        model_version=model_loader.model_version(),
    )


def _require_model() -> None:
    if not model_loader.is_loaded() and not model_loader.try_load():
        raise HTTPException(
            status_code=503,
            detail="Model is not loaded. Check MODEL_URI/MODEL_PATH configuration.",
        )


@app.post("/predict", response_model=PredictionResponse, tags=["inference"])
def predict(request: DeliveryRequest) -> PredictionResponse:
    """Estimate delivery time for a single order."""
    _require_model()
    try:
        preds = model_loader.predict_frame([request.model_dump()])
    except Exception as exc:  # noqa: BLE001 - surface as a clean 400
        logger.exception("Prediction failed")
        raise HTTPException(
            status_code=400, detail=f"Prediction failed: {exc}"
        ) from exc
    return PredictionResponse(
        predicted_time_min=preds[0],
        model_version=model_loader.model_version() or "unknown",
    )


@app.post("/predict/batch", response_model=BatchPredictionResponse, tags=["inference"])
def predict_batch(request: BatchDeliveryRequest) -> BatchPredictionResponse:
    """Estimate delivery time for many orders in one call."""
    _require_model()
    try:
        preds = model_loader.predict_frame([r.model_dump() for r in request.instances])
    except Exception as exc:  # noqa: BLE001
        logger.exception("Batch prediction failed")
        raise HTTPException(
            status_code=400, detail=f"Prediction failed: {exc}"
        ) from exc
    return BatchPredictionResponse(
        predictions=preds,
        model_version=model_loader.model_version() or "unknown",
        count=len(preds),
    )


if __name__ == "__main__":
    import os

    import uvicorn

    uvicorn.run(
        "food_on_the_fly.api.main:app",
        host="0.0.0.0",
        port=int(os.getenv("PORT", "8080")),
    )
