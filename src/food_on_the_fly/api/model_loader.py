"""Model loading and prediction helpers for the API.

Loading strategy (first that works wins):

1. ``MODEL_URI`` env var, loaded via MLflow. Accepts any MLflow URI:
   ``models:/food_on_the_fly/Staging``, ``runs:/<id>/model``,
   ``gs://bucket/path``, or a local MLmodel directory.
2. ``MODEL_PATH`` env var pointing at a local joblib/pickle artifact.
3. Default ``MODEL_URI`` of ``models:/food_on_the_fly/Staging`` so a
   registered Staging model is picked up automatically in the cloud.

Loading is lazy and fault-tolerant: if no model is available the API
still boots and serves ``/health`` (reporting ``model_loaded=false``)
while ``/predict`` returns 503. This keeps Cloud Run deploys healthy
and makes a missing-model condition observable rather than crash-looping.
"""

from __future__ import annotations

import os
import threading
from typing import Any

import pandas as pd

from food_on_the_fly.logging_config import get_logger

logger = get_logger(__name__)

_DEFAULT_MODEL_URI = "models:/food_on_the_fly/Staging"

_model: Any = None
_model_version: str | None = None
_lock = threading.Lock()


def _load() -> tuple[Any, str | None]:
    """Attempt to load the model. Returns (model, version) or (None, None)."""
    model_path = os.getenv("MODEL_PATH")
    if model_path:
        import joblib

        logger.info("Loading model from local path %s", model_path)
        return joblib.load(model_path), os.path.basename(model_path)

    model_uri = os.getenv("MODEL_URI", _DEFAULT_MODEL_URI)
    import mlflow.pyfunc

    logger.info("Loading model from MLflow URI %s", model_uri)
    return mlflow.pyfunc.load_model(model_uri), model_uri


def get_model() -> tuple[Any, str | None]:
    """Return the cached model, loading it on first use. Thread-safe."""
    global _model, _model_version
    if _model is None:
        with _lock:
            if _model is None:
                _model, _model_version = _load()
    return _model, _model_version


def try_load() -> bool:
    """Eagerly load the model; log and swallow failures. Returns success."""
    try:
        model, _ = get_model()
        return model is not None
    except Exception as exc:  # noqa: BLE001 - boot must survive any load error
        logger.warning("Model could not be loaded at startup: %s", exc)
        return False


def is_loaded() -> bool:
    return _model is not None


def model_version() -> str | None:
    return _model_version


def predict_frame(rows: list[dict[str, Any]]) -> list[float]:
    """Score a list of raw records and return predictions in minutes."""
    model, _ = get_model()
    frame = pd.DataFrame(rows)
    preds = model.predict(frame)
    # MLflow pyfunc may return a DataFrame/Series; normalise to a flat list.
    if hasattr(preds, "to_numpy"):
        preds = preds.to_numpy()
    flat = [float(p) for p in (preds.ravel() if hasattr(preds, "ravel") else preds)]
    # Delivery time cannot be negative; clamp defensively.
    return [max(0.0, round(p, 2)) for p in flat]
