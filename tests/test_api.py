"""API tests using a lightweight stub pipeline (no MLflow required).

We inject a fake model into ``model_loader`` so the request -> DataFrame
-> prediction path is exercised in CI without a registered model.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from fastapi.testclient import TestClient

from food_on_the_fly.api import main, model_loader
from food_on_the_fly.api.schemas import _EXAMPLE


class _StubModel:
    """Returns a constant per row; records the columns it received."""

    seen_columns: list[str] | None = None

    def predict(self, frame: pd.DataFrame) -> np.ndarray:
        _StubModel.seen_columns = list(frame.columns)
        return np.full(len(frame), 25.0)


def _install_stub() -> None:
    model_loader._model = _StubModel()
    model_loader._model_version = "stub-1.0"


def test_health_without_model() -> None:
    model_loader._model = None
    model_loader._model_version = None
    client = TestClient(main.app)
    # TestClient context triggers lifespan; default URI load will fail -> degraded.
    body = client.get("/health").json()
    assert body["status"] == "ok"
    assert "model_loaded" in body


def test_predict_single() -> None:
    _install_stub()
    client = TestClient(main.app)
    resp = client.post("/predict", json=_EXAMPLE)
    assert resp.status_code == 200, resp.text
    data = resp.json()
    assert data["predicted_time_min"] == 25.0
    assert data["model_version"] == "stub-1.0"
    # All 16 raw feature columns must reach the pipeline.
    assert set(_EXAMPLE).issubset(set(_StubModel.seen_columns or []))


def test_predict_batch() -> None:
    _install_stub()
    client = TestClient(main.app)
    resp = client.post("/predict/batch", json={"instances": [_EXAMPLE, _EXAMPLE]})
    assert resp.status_code == 200, resp.text
    data = resp.json()
    assert data["count"] == 2
    assert data["predictions"] == [25.0, 25.0]


def test_predict_negative_clamped() -> None:
    class _NegModel:
        def predict(self, frame: pd.DataFrame) -> np.ndarray:
            return np.full(len(frame), -5.0)

    model_loader._model = _NegModel()
    model_loader._model_version = "neg"
    client = TestClient(main.app)
    resp = client.post("/predict", json=_EXAMPLE)
    assert resp.json()["predicted_time_min"] == 0.0
