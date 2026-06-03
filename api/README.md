# API Directory

The FastAPI inference service lives in the installable package at
[`src/food_on_the_fly/api/`](../src/food_on_the_fly/api/):

- **`main.py`** — FastAPI app (`/`, `/health`, `/predict`, `/predict/batch`)
- **`schemas.py`** — Pydantic request/response models (raw Zomato columns)
- **`model_loader.py`** — loads the trained pipeline from an MLflow URI
  (`MODEL_URI`, default `models:/food_on_the_fly/Staging`) or a local
  joblib artifact (`MODEL_PATH`)

## Run locally

```bash
# Against a registered Staging model (needs MLflow tracking/registry):
uvicorn food_on_the_fly.api.main:app --reload --port 8080

# Against a local artifact:
MODEL_PATH=models/model.joblib uvicorn food_on_the_fly.api.main:app --port 8080
```

Then open http://localhost:8080/docs for interactive Swagger docs.

## Container

Built and deployed by `.github/workflows/deploy.yml` using
[`dockerfiles/Dockerfile.api`](../dockerfiles/Dockerfile.api), which runs
uvicorn on `$PORT` (Cloud Run sets `PORT=8080`).
