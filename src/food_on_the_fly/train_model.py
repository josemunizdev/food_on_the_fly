"""Model training entrypoint."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import hydra
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
import xgboost as xgb
from omegaconf import DictConfig, OmegaConf
from sklearn.compose import ColumnTransformer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler

from food_on_the_fly.data.loaders import load_processed
from food_on_the_fly.features.build_features import (
    create_day_of_week_transformer,
    create_haversine_transformer,
    # create_rush_hour_transformer,
    # create_weekday_transformer,
    create_hour_of_day_transformer,
)
from food_on_the_fly.logging_config import get_logger, setup_logging
from food_on_the_fly.utils.monitoring import SystemMetricsLogger
from food_on_the_fly.utils.seed import set_seed

logger = get_logger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = str(PROJECT_ROOT / "configs")


def make_categorical_pipeline(transformer: Any, transformer_name: str) -> Pipeline:

    def df_to_array(X: pd.DataFrame) -> np.ndarray:
        return X.values.reshape(-1, 1)

    return Pipeline(
        [
            (transformer_name, transformer),
            ("to_array", FunctionTransformer(df_to_array, validate=False)),
            ("encoder", OneHotEncoder(drop="first", handle_unknown="ignore")),
        ]
    )


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
        # Enable system metrics logging
        mlflow.enable_system_metrics_logging()
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
            # "Time_Orderd",
            # "Order_Date",
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
        # Create HaversineTransformer from config

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
        haversine_transformer = create_haversine_transformer(cfg)
        # weekday_transformer = create_weekday_transformer(cfg)
        # rush_hour_transformer = create_rush_hour_transformer(cfg)
        hour_of_day_transformer = create_hour_of_day_transformer(cfg)
        day_of_week_transformer = create_day_of_week_transformer(cfg)

        # Build column transformer
        preprocessor = ColumnTransformer(
            transformers=[
                ("distance", haversine_transformer, location_features),
                # ("weekday", weekday_transformer, ["Order_Date"]),
                # ("rush_hour", rush_hour_transformer, ["Time_Orderd"]),
                (
                    "hour",
                    make_categorical_pipeline(hour_of_day_transformer, "hour_of_day"),
                    ["Time_Orderd"],
                ),
                (
                    "day",
                    make_categorical_pipeline(day_of_week_transformer, "day_of_week"),
                    ["Order_Date"],
                ),
                ("numeric", StandardScaler(), numeric_features),
                (
                    "categorical",
                    OneHotEncoder(drop="first", handle_unknown="ignore"),
                    categorical_features,
                ),
            ],
            remainder="drop",
            verbose_feature_names_out=True,
        )

        # Complete pipeline: preprocessing + XGBoost
        pipeline = Pipeline(
            [
                ("preprocessor", preprocessor),
                ("regressor", xgb.XGBRegressor(**cfg.model.parameters)),
            ]
        )

        # Train the model with system-metrics monitoring
        logger.info("Training XGBoost model...")
        train_start = time.perf_counter()
        with SystemMetricsLogger(interval_seconds=0.1):
            pipeline.fit(X_train, y_train)
        train_duration = time.perf_counter() - train_start
        mlflow.log_metric("train_duration_seconds", train_duration)
        logger.info("Training complete in %.2fs", train_duration)

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

        import matplotlib

        matplotlib.use("Agg")  # Use non-interactive backend for plotting
        import matplotlib.pyplot as plt

        logger.info("Generating CML outputs...")
        with open("metrics.txt", "w") as f:
            f.write("=== TRAINING METRICS ===\n")
            f.write(f"RMSE: {train_rmse:.2f} minutes\n")
            f.write(f"MAE:  {train_mae:.2f} minutes\n")
            f.write(f"R²:   {train_r2:.4f}\n\n")
            f.write("=== VALIDATION METRICS ===\n")
            f.write(f"RMSE: {val_rmse:.2f} minutes\n")
            f.write(f"MAE:  {val_mae:.2f} minutes\n")
            f.write(f"R²:   {val_r2:.4f}\n")

        plt.figure(figsize=(10, 6))
        plt.scatter(y_val, y_val_pred, alpha=0.5, s=10)
        plt.plot(
            [y_val.min(), y_val.max()],
            [y_val.min(), y_val.max()],
            "r--",
            lw=2,
            label="Perfect Prediction",
        )
        plt.xlabel("Actual Delivery Time (min)", fontsize=12)
        plt.ylabel("Predicted Delivery Time (min)", fontsize=12)
        plt.title("Prediction vs Actual Delivery Time", fontsize=14)
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig("prediction_scatter.png", dpi=100, bbox_inches="tight")
        plt.close()

        # 3. Residual plot
        residuals = y_val - y_val_pred
        plt.figure(figsize=(10, 6))
        plt.scatter(y_val_pred, residuals, alpha=0.5, s=10)
        plt.axhline(y=0, color="r", linestyle="--", lw=2)
        plt.xlabel("Predicted Delivery Time (min)", fontsize=12)
        plt.ylabel("Residuals (Actual - Predicted)", fontsize=12)
        plt.title("Residual Plot - Are Errors Random?", fontsize=14)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig("residual_plot.png", dpi=100, bbox_inches="tight")
        plt.close()

        # 4. Error distribution histogram
        plt.figure(figsize=(10, 6))
        plt.hist(residuals, bins=50, edgecolor="black", alpha=0.7)
        plt.xlabel("Prediction Error (minutes)", fontsize=12)
        plt.ylabel("Frequency", fontsize=12)
        plt.title("Distribution of Prediction Errors", fontsize=14)
        plt.axvline(x=0, color="r", linestyle="--", lw=2, label="Zero Error")
        plt.legend()
        plt.grid(True, alpha=0.3, axis="y")
        plt.tight_layout()
        plt.savefig("error_distribution.png", dpi=100, bbox_inches="tight")
        plt.close()

        # 5. Feature importance (XGBoost specific)
        xgb_model = pipeline.named_steps["regressor"]
        importance_dict = xgb_model.get_booster().get_score(importance_type="weight")

        if importance_dict:
            # Get feature names from the preprocessor
            try:
                feature_names = pipeline.named_steps[
                    "preprocessor"
                ].get_feature_names_out()
            except AttributeError:
                # Fallback if get_feature_names_out() not available
                feature_names = [f"f{i}" for i in range(len(importance_dict))]

            # Map f0, f1, f2... to actual feature names
            importance_with_names = {}
            for f_idx, importance in importance_dict.items():
                # f_idx is like 'f0', 'f1', etc.
                idx = int(f_idx[1:])  # Extract the number
                if idx < len(feature_names):
                    feature_name = feature_names[idx]
                    # Clean up the name (remove transformer prefixes)
                    if "__" in feature_name:
                        feature_name = feature_name.split("__", 1)[1]
                    importance_with_names[feature_name] = importance

            # Sort by importance
            sorted_importance = sorted(
                importance_with_names.items(), key=lambda x: x[1], reverse=True
            )[:15]  # Top 15 features

            features = [item[0] for item in sorted_importance]
            importances = [item[1] for item in sorted_importance]

            plt.figure(figsize=(10, 8))
            plt.barh(range(len(features)), importances)
            plt.yticks(range(len(features)), features, fontsize=10)
            plt.xlabel("Importance (Weight)", fontsize=12)
            plt.ylabel("Feature", fontsize=12)
            plt.title("Top 15 Feature Importances", fontsize=14)
            plt.tight_layout()
            plt.savefig("feature_importance.png", dpi=100, bbox_inches="tight")
            plt.close()

            logger.info(f"Top 5 important features: {features[:5]}")

        logger.info("CML outputs generated: metrics.txt + 4 PNG files")

        # Log model to MLflow
        mlflow.sklearn.log_model(pipeline, "model")
        logger.info("Model logged to MLflow")

        logger.info("Training complete! 🎉")


if __name__ == "__main__":
    main()
