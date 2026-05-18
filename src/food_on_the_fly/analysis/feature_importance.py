from __future__ import annotations
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import mlflow
import numpy as np
import pandas as pd
import seaborn as sns

from food_on_the_fly.logging_config import get_logger

logger = get_logger(__name__)

def extract_feature_names(pipeline: Any) -> list[str]:
    """Extract feature names from a fitted pipeline."""
    preprocessor = pipeline.named_steps["preprocessor"]
    feature_names = preprocessor.get_feature_names_out()
    return list(feature_names)

def get_feature_importance(
    pipeline: Any,
    importance_type: str = "gain",
) -> pd.DataFrame:
    """Extract feature importance from the XGBoost model in the pipeline."""
    logger.info("Extracting feature importance using %s", importance_type)
    xgb_model = pipeline.named_steps["regressor"]
    
    feature_names = extract_feature_names(pipeline)
    
    importance_dict = xgb_model.get_booster().get_score(importance_type=importance_type)
    
    importances= []
    
    for i, feature in enumerate(feature_names):
        feature_key = f"f{i}"
        importance = importance_dict.get(feature_key, 0.0)
        importances.append({
            "feature": feature,
            "importance": importance,
        })
    df_importance = pd.DataFrame(importances)
    df_importance = df_importance.sort_values(by="importance", ascending=False).reset_index(drop=True)
    
    logger.info("Top 10 features by importance:\n%s", df_importance.head(10))
    return df_importance


def plot_top_features(
    df_importance: pd.DataFrame,
    top_n: int = 20,
    figsize: tuple[int, int] = (10, 8),
    save_path: Path | None = None,
) -> None:
    top_features = df_importance.head(top_n)
    
    plt.figure(figsize=figsize)
    sns.barplot(x="importance", y="feature", data=top_features, palette="viridis")
    plt.title(f"Top {top_n} Feature Importances")
    plt.xlabel("Importance")
    plt.ylabel("Feature")
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
        logger.info("Saved feature importance plot to %s", save_path)
    plt.show()
    
    logger.info("Plotted top %d features", top_n)
    
    for idx, row in top_features.iterrows():
        logger.info("Feature: %s, Importance: %.4f", row["feature"], row["importance"]) 
        
def main(
    run_id:str | None = None,
    model_path: Path | None = None,
    importance_type: str = "gain",
    top_n: int = 20,
    save_plot: bool = True,
)-> pd.DataFrame:
    if run_id:
        logger.info(f"Loading model from MLflow run: {run_id}")
        model_uri = f"runs:/{run_id}/model"
        pipeline = mlflow.sklearn.load_model(model_uri)
    elif model_path:
        logger.info(f"Loading model from path: {model_path}")
        import joblib
        pipeline = joblib.load(model_path)
    else:
        raise ValueError("Must provide either run_id or model_path")
    df_importance = get_feature_importance(pipeline, importance_type)
    if save_plot:
        from food_on_the_fly.config import FIGURES_DIR
        plot_path = FIGURES_DIR / f"feature_importance_top{top_n}.png"
        
    else:
        plot_path = None
    plot_top_features(df_importance, top_n=top_n, save_path=plot_path)
    return df_importance

if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python feature_importance.py <mlflow_run_id>")
        print("   or: python feature_importance.py --model-path <path>")
        sys.exit(1)

    if sys.argv[1] == "--model-path":
        model_path = Path(sys.argv[2])
        df_importance = main(model_path=model_path)
    else:
        run_id = sys.argv[1]
        df_importance = main(run_id=run_id)

    # Optionally save to CSV
    from food_on_the_fly.config import REPORTS_DIR
    output_path = REPORTS_DIR / "feature_importances.csv"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df_importance.to_csv(output_path, index=False)
    print(f"\nFeature importances saved to: {output_path}")
    