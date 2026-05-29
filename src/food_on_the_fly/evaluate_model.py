"""Evaluate trained model on baseline test set."""

from __future__ import annotations

import time
from pathlib import Path

import hydra
import mlflow
import mlflow.sklearn
import numpy as np
from omegaconf import DictConfig
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from food_on_the_fly.data.loaders import load_processed
from food_on_the_fly.logging_config import get_logger, setup_logging
from food_on_the_fly.utils.seed import set_seed

logger = get_logger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = str(PROJECT_ROOT / "configs")


@hydra.main(version_base=None, config_path=CONFIG_DIR, config_name="config")
def main(cfg: DictConfig) -> None:
    """Evaluate model on baseline test set.

    Args:
        cfg (DictConfig): Configuration object from Hydra
    """
    setup_logging()
    set_seed(cfg.data.random_state)

    logger.info("=" * 80)
    logger.info("BASELINE TEST SET EVALUATION")
    logger.info("=" * 80)

    mlflow.set_tracking_uri(cfg.mlflow.tracking_uri)
    mlflow.set_experiment(cfg.mlflow.experiment_name)

    # Load model based on selection mode
    client = mlflow.tracking.MlflowClient()
    experiment = client.get_experiment_by_name(cfg.mlflow.experiment_name)
    selection_mode = cfg.evaluation.selection_mode
    logger.info(f"Model selection mode: {selection_mode}")

    if selection_mode == "specific":
        if not cfg.evaluation.run_id:
            logger.error(
                "Selection mode is 'specific' but no run_id provided in config!"
            )
            return
        run_id = cfg.evaluation.run_id
        logger.info(f"Loading model from specified run_id: {run_id}")
        latest_run = client.get_run(run_id)

    elif selection_mode == "best":
        # Find best model by validation metric
        metric_name = cfg.evaluation.best_metric
        metric_order = cfg.evaluation.best_metric_order

        logger.info(f"Finding best model by metric: {metric_name} ({metric_order})")

        all_runs = client.search_runs(
            experiment_ids=[experiment.experiment_id],
            order_by=[
                "start_time DESC"
            ],  # Get latest runs first to break ties by recency
            max_results=100,  # limit 100
        )
        runs_with_metric = [run for run in all_runs if metric_name in run.data.metrics]
        if not runs_with_metric:
            logger.error(
                f"No runs found with metric '{metric_name}'! Train a model first."
            )
            return
        logger.info(f"Found {len(runs_with_metric)} runs with {metric_name}")

        # Sort by metric value
        if metric_order == "min":  # For RMSE, MAE (lower is better)
            best_run = min(runs_with_metric, key=lambda r: r.data.metrics[metric_name])
        else:
            # For R2, accuracy (higher is better)
            best_run = max(runs_with_metric, key=lambda r: r.data.metrics[metric_name])
        latest_run = best_run
        run_id = latest_run.info.run_id
        best_metric_value = latest_run.data.metrics[metric_name]
        logger.info(f"Best model: {run_id}")
        logger.info(f"  {metric_name}: {best_metric_value:.4f}")
        logger.info(f"  Run name: {latest_run.data.tags.get('mlflow.runName', 'N/A')}")

    else:
        # Default to latest run
        logger.info("Finding latest run...")
        all_runs = client.search_runs(
            experiment_ids=[experiment.experiment_id],
            order_by=["start_time DESC"],
            max_results=50,
        )
        training_runs = [
            run
            for run in all_runs
            if not run.data.tags.get("mlflow.runName", "").startswith("eval_")
        ]

        if not training_runs:
            logger.error("No trained models found! Train a model first.")
            return

        latest_run = training_runs[0]
        run_id = latest_run.info.run_id
        logger.info(f"Selected run: {run_id}")
        logger.info(f"Run name: {latest_run.data.tags.get('mlflow.runName', 'N/A')}")

    # Load model
    model_uri = f"runs:/{run_id}/model"
    model = mlflow.sklearn.load_model(model_uri)
    logger.info("Model loaded successfully")

    # Load test data
    logger.info("Loading baseline test set...")
    test_df = load_processed("test.csv")
    logger.info(f"Test samples: {len(test_df)}")

    # Separate features and target
    target_col = "Time_taken (min)"
    drop_cols = [target_col, "ID", "Delivery_person_ID"]
    X_test = test_df.drop(columns=drop_cols)
    y_test = test_df[target_col]

    # Start MLflow run for evaluation
    with mlflow.start_run(run_name=f"eval_baseline_test_{run_id[:8]}"):
        # Log parent run info
        mlflow.log_param("parent_run_id", run_id)
        mlflow.log_param(
            "parent_run_name", latest_run.data.tags.get("mlflow.runName", "N/A")
        )
        mlflow.log_param("evaluation_type", "baseline_test")
        mlflow.log_param("test_samples", len(test_df))

        # Copy ALL hyperparameters from parent training run
        logger.info("Copying hyperparameters from training run...")
        parent_params = latest_run.data.params
        for param_name, param_value in parent_params.items():
            mlflow.log_param(f"model_{param_name}", param_value)
        logger.info(f"Copied {len(parent_params)} hyperparameters to evaluation run")
        # Make predictions
        logger.info("Making predictions on test set...")
        eval_start = time.perf_counter()
        y_pred = model.predict(X_test)
        eval_duration = time.perf_counter() - eval_start

        # Copy key validation metrics for comparison
        parent_metrics = latest_run.data.metrics
        if "val_rmse" in parent_metrics:
            mlflow.log_metric("parent_val_rmse", parent_metrics["val_rmse"])
        if "val_mae" in parent_metrics:
            mlflow.log_metric("parent_val_mae", parent_metrics["val_mae"])
        if "val_r2" in parent_metrics:
            mlflow.log_metric("parent_val_r2", parent_metrics["val_r2"])

        # Calculate metrics
        test_mse = mean_squared_error(y_test, y_pred)
        test_rmse = np.sqrt(test_mse)
        test_mae = mean_absolute_error(y_test, y_pred)
        test_r2 = r2_score(y_test, y_pred)

        # Log metrics to MLflow
        mlflow.log_metric("test_mse", test_mse)
        mlflow.log_metric("test_rmse", test_rmse)
        mlflow.log_metric("test_mae", test_mae)
        mlflow.log_metric("test_r2", test_r2)
        mlflow.log_metric("eval_duration_seconds", eval_duration)

        if "val_rmse" in parent_metrics:
            val_rmse = parent_metrics["val_rmse"]
            rmse_degradation = test_rmse - val_rmse
            rmse_degradation_pct = (test_rmse / val_rmse - 1) * 100
            mlflow.log_metric("rmse_degradation", rmse_degradation)
            mlflow.log_metric("rmse_degradation_pct", rmse_degradation_pct)

            logger.info(f"Validation RMSE: {val_rmse:.2f} minutes")
            logger.info(f"Test RMSE: {test_rmse:.2f} minutes")
            logger.info(
                f"Degradation:{rmse_degradation:+.2f}minutes({rmse_degradation_pct:+.1f}%)"
            )

            if abs(rmse_degradation_pct) > 10:
                logger.warning("⚠️  Test performance differs from validation by >10%!")
            else:
                logger.info(
                    "✅ Test performance matches validation (good generalization)"
                )

        # Log to console
        logger.info("=" * 80)
        logger.info("BASELINE TEST SET RESULTS")
        logger.info(f"  RMSE: {test_rmse:.2f} minutes")
        logger.info(f"  MAE:  {test_mae:.2f} minutes")
        logger.info(f"  R²:   {test_r2:.4f}")
        logger.info(f"  Evaluation time:{eval_duration:.2f}s")
        logger.info("=" * 80)
        logger.info("Results logged to MLflow")
        logger.info("Baseline evaluation complete! 🎉")


if __name__ == "__main__":
    main()
