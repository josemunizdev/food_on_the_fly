"""Model training entrypoint."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import hydra
import mlflow
import mlflow.sklearn
import numpy as np
import xgboost as xgb
from omegaconf import DictConfig, OmegaConf
from sklearn.compose import ColumnTransformer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from food_on_the_fly.data.loaders import load_processed
from food_on_the_fly.features.build_features import create_haversine_transformer
from food_on_the_fly.logging_config import get_logger, setup_logging
from food_on_the_fly.utils.seed import set_seed

logger = get_logger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = str(PROJECT_ROOT / "configs")


@hydra.main(version_base=None, config_path=CONFIG_DIR, config_name="config")
def main(cfg: DictConfig) -> None:
    """Train XGBoost model with MLflow tracking
    Args:
        cfg (DictConfig): Configuration object from Hydra
    """
    setup_logging()
    set_seed(cfg.data.random_state)
    logger.info("Starting training with config:\n%s", OmegaConf.to_yaml(cfg))
    logger.info("Using Hydra + MLflow + XGBoost for training")
    logger.info("=" * 80)
    logger.info("Configurations")
    logger.info(OmegaConf.to_yaml(cfg))

    mlflow.set_tracking_uri(cfg.mlflow.tracking_uri)
    mlflow.set_experiment(cfg.mlflow.experiment_name)

    with mlflow.start_run(run_name=cfg.mlflow.run_name):
        # Log config parameters to MLflow (with type safety)
        params = OmegaConf.to_container(cfg, resolve=True)
        if isinstance(params, dict):
            # Ensure all keys are strings as required by MLflow
            string_params: dict[str, Any] = {str(k): v for k, v in params.items()}
            mlflow.log_params(string_params)
        else:
            logger.warning("Config could not be converted to dict format for MLflow")

        # Load pre-split data
        logger.info("Loading pre-split training and validation data...")
        train_df = load_processed("train.csv")
        val_df = load_processed("val.csv")

        logger.info(
            f"Train samples: {len(train_df)}, Validation samples: {len(val_df)}"
        )

        # Separate features and target
        target_col = "Time_taken (min)"
        # Drop target and ID columns
        drop_cols = [
            target_col,
            "ID",
            "Delivery_person_ID",
        ]
        X_train = train_df.drop(columns=drop_cols)
        y_train = train_df[target_col]
        logger.info(
            "Features and target separated. X shape: %s, y shape: %s",
            X_train.shape,
            y_train.shape,
        )
        logger.info(f"Target column: {target_col}")

        # Separate features and target for validation set
        X_val = val_df.drop(columns=drop_cols)
        y_val = val_df[target_col]

        logger.info(f"Train samples: {len(X_train)}, Validation samples: {len(X_val)}")

        # Build preprocessing pipeline
        logger.info("Building preprocessing pipeline...")

        # Define column types
        numeric_features = [
            "Delivery_person_Age",
            "Delivery_person_Ratings",
            "Vehicle_condition",
            "multiple_deliveries",
        ]

        categorical_features = [
            "Weather_conditions",
            "Road_traffic_density",
            "Type_of_order",
            "Type_of_vehicle",
            "Festival",
            "City",
        ]

        location_features = [
            "Restaurant_latitude",
            "Restaurant_longitude",
            "Delivery_location_latitude",
            "Delivery_location_longitude",
        ]

        # Create HaversineTransformer from config
        haversine_transformer = create_haversine_transformer(cfg)

        # Build column transformer
        preprocessor = ColumnTransformer(
            transformers=[
                ("distance", haversine_transformer, location_features),
                ("numeric", StandardScaler(), numeric_features),
                (
                    "categorical",
                    OneHotEncoder(drop="first", handle_unknown="ignore"),
                    categorical_features,
                ),
            ],
            remainder="drop",
        )

        # Complete pipeline: preprocessing + XGBoost
        pipeline = Pipeline(
            [
                ("preprocessor", preprocessor),
                ("regressor", xgb.XGBRegressor(**cfg.model.parameters)),
            ]
        )

        # Train the model
        logger.info("Training XGBoost model...")
        pipeline.fit(X_train, y_train)
        logger.info("Training complete!")

        # Make predictions
        logger.info("Making predictions...")
        y_train_pred = pipeline.predict(X_train)
        y_val_pred = pipeline.predict(X_val)

        # Calculate metrics
        train_mse = mean_squared_error(y_train, y_train_pred)
        train_rmse = np.sqrt(train_mse)
        train_mae = mean_absolute_error(y_train, y_train_pred)
        train_r2 = r2_score(y_train, y_train_pred)

        val_mse = mean_squared_error(y_val, y_val_pred)
        val_rmse = np.sqrt(val_mse)
        val_mae = mean_absolute_error(y_val, y_val_pred)
        val_r2 = r2_score(y_val, y_val_pred)

        # Log metrics to MLflow
        mlflow.log_metric("train_mse", train_mse)
        mlflow.log_metric("train_rmse", train_rmse)
        mlflow.log_metric("train_mae", train_mae)
        mlflow.log_metric("train_r2", train_r2)
        mlflow.log_metric("val_mse", val_mse)
        mlflow.log_metric("val_rmse", val_rmse)
        mlflow.log_metric("val_mae", val_mae)
        mlflow.log_metric("val_r2", val_r2)

        # Log to console
        logger.info("=" * 80)
        logger.info("TRAINING METRICS")
        logger.info(f"  RMSE: {train_rmse:.2f} minutes")
        logger.info(f"  MAE:  {train_mae:.2f} minutes")
        logger.info(f"  R²:   {train_r2:.4f}")
        logger.info("")
        logger.info("VALIDATION METRICS")
        logger.info(f"  RMSE: {val_rmse:.2f} minutes")
        logger.info(f"  MAE:  {val_mae:.2f} minutes")
        logger.info(f"  R²:   {val_r2:.4f}")
        logger.info("=" * 80)

        # Log model to MLflow
        mlflow.sklearn.log_model(pipeline, "model")
        logger.info("Model logged to MLflow")

        logger.info("Training complete! 🎉")


if __name__ == "__main__":
    main()
