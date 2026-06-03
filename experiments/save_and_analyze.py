"""
Save and analyze transformer comparison results.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

# Your results data (paste the dataframe here)
results_data = {
    "new_model_optim": {
        "sys_cpu_percent": 54.2,
        "sys_mem_percent": 79.3,
        "sys_proc_rss_mb": 283.9,
        "train_duration_seconds": 0.514,
        "train_mae": 2.572,
        "train_mse": 10.49,
        "train_r2": 0.879,
        "train_rmse": 3.239,
        "val_mae": 3.178,
        "val_mse": 16.24,
        "val_r2": 0.817,
        "val_rmse": 4.03,
        "eval_duration_seconds": 0.014,
        "rmse_degradation": -0.048,
        "rmse_degradation_pct": -1.191,
        "test_mae": 3.132,
        "test_mse": 15.85,
        "test_r2": 0.822,
        "test_rmse": 3.982,
    },
    "new_model_fast": {
        "sys_cpu_percent": 98.2,
        "sys_mem_percent": 77.1,
        "sys_proc_rss_mb": 278.6,
        "train_duration_seconds": 0.328,
        "train_mae": 3.717,
        "train_mse": 21.82,
        "train_r2": 0.748,
        "train_rmse": 4.672,
        "val_mae": 3.796,
        "val_mse": 23.26,
        "val_r2": 0.737,
        "val_rmse": 4.823,
        "eval_duration_seconds": 0.019,
        "rmse_degradation": -0.124,
        "rmse_degradation_pct": -2.566,
        "test_mae": 3.725,
        "test_mse": 22.08,
        "test_r2": 0.752,
        "test_rmse": 4.699,
    },
    "new_model_baseline": {
        "train_duration_seconds": 0.077,
        "train_mae": 3.917,
        "train_mse": 24.47,
        "train_r2": 0.717,
        "train_rmse": 4.947,
        "val_mae": 4.004,
        "val_mse": 26.02,
        "val_r2": 0.706,
        "val_rmse": 5.101,
        "eval_duration_seconds": 0.015,
        "rmse_degradation": -0.127,
        "rmse_degradation_pct": -2.497,
        "test_mae": 3.93,
        "test_mse": 24.73,
        "test_r2": 0.722,
        "test_rmse": 4.973,
    },
    "old_model_optim": {
        "sys_cpu_percent": 0,
        "sys_mem_percent": 77.8,
        "sys_proc_rss_mb": 277.9,
        "train_duration_seconds": 0.489,
        "train_mae": 2.621,
        "train_mse": 10.89,
        "train_r2": 0.874,
        "train_rmse": 3.3,
        "val_mae": 3.17,
        "val_mse": 16.14,
        "val_r2": 0.818,
        "val_rmse": 4.017,
        "eval_duration_seconds": 0.016,
        "rmse_degradation": -0.046,
        "rmse_degradation_pct": -1.138,
        "test_mae": 3.129,
        "test_mse": 15.77,
        "test_r2": 0.823,
        "test_rmse": 3.972,
    },
    "old_model_fast": {
        "sys_cpu_percent": 69.2,
        "sys_mem_percent": 77.8,
        "sys_proc_rss_mb": 281.6,
        "train_duration_seconds": 0.214,
        "train_mae": 3.717,
        "train_mse": 21.79,
        "train_r2": 0.748,
        "train_rmse": 4.668,
        "val_mae": 3.798,
        "val_mse": 23.2,
        "val_r2": 0.738,
        "val_rmse": 4.817,
        "eval_duration_seconds": 0.015,
        "rmse_degradation": -0.121,
        "rmse_degradation_pct": -2.515,
        "test_mae": 3.735,
        "test_mse": 22.05,
        "test_r2": 0.752,
        "test_rmse": 4.696,
    },
    "old_model_baseline": {
        "train_duration_seconds": 0.08,
        "train_mae": 3.909,
        "train_mse": 24.39,
        "train_r2": 0.718,
        "train_rmse": 4.938,
        "val_mae": 3.996,
        "val_mse": 25.91,
        "val_r2": 0.707,
        "val_rmse": 5.09,
        "eval_duration_seconds": 0.013,
        "rmse_degradation": -0.127,
        "rmse_degradation_pct": -2.499,
        "test_mae": 3.919,
        "test_mse": 24.63,
        "test_r2": 0.723,
        "test_rmse": 4.963,
    },
}


def save_results() -> pd.DataFrame:
    """Save results to CSV and Excel."""
    df = pd.DataFrame(results_data)

    output_dir = Path("experiments/results")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save to CSV
    csv_file = output_dir / "transformer_comparison_results.csv"
    df.to_csv(csv_file)
    print(f"✅ Saved CSV: {csv_file}")

    # Save to Excel with formatting
    excel_file = output_dir / "transformer_comparison_results.xlsx"
    df.to_excel(excel_file)
    print(f"✅ Saved Excel: {excel_file}")

    return df


def create_comparison_table(df: pd.DataFrame) -> pd.DataFrame:
    """Create clean comparison table."""
    # Parse column names
    parsed = []
    for col in df.columns:
        parts = col.split("_")
        transformer = parts[0]  # 'old' or 'new'
        config = "_".join(parts[2:])  # 'baseline','fast', 'optim'
        parsed.append({"col": col, "transformer": transformer, "config": config})

    # Create comparison for key metrics
    comparison = pd.DataFrame(
        {
            "Config": ["Baseline", "Fast", "Optimized"],
            "OLD Test RMSE": [
                df["old_model_baseline"]["test_rmse"],
                df["old_model_fast"]["test_rmse"],
                df["old_model_optim"]["test_rmse"],
            ],
            "NEW Test RMSE": [
                df["new_model_baseline"]["test_rmse"],
                df["new_model_fast"]["test_rmse"],
                df["new_model_optim"]["test_rmse"],
            ],
            "OLD Test R²": [
                df["old_model_baseline"]["test_r2"],
                df["old_model_fast"]["test_r2"],
                df["old_model_optim"]["test_r2"],
            ],
            "NEW Test R²": [
                df["new_model_baseline"]["test_r2"],
                df["new_model_fast"]["test_r2"],
                df["new_model_optim"]["test_r2"],
            ],
            "OLD Train Time (s)": [
                df["old_model_baseline"]["train_duration_seconds"],
                df["old_model_fast"]["train_duration_seconds"],
                df["old_model_optim"]["train_duration_seconds"],
            ],
            "NEW Train Time (s)": [
                df["new_model_baseline"]["train_duration_seconds"],
                df["new_model_fast"]["train_duration_seconds"],
                df["new_model_optim"]["train_duration_seconds"],
            ],
        }
    )

    # Calculate differences
    comparison["RMSE Diff"] = comparison["NEW Test RMSE"] - comparison["OLD Test RMSE"]
    comparison["RMSE Diff %"] = (
        comparison["RMSE Diff"] / comparison["OLD Test RMSE"] * 100
    )
    comparison["R² Diff"] = comparison["NEW Test R²"] - comparison["OLD Test R²"]

    # Save
    output_dir = Path("experiments/results")
    comparison.to_csv(output_dir / "comparison_summary.csv", index=False)
    print(f"\n✅ Saved comparison: {output_dir / 'comparison_summary.csv'}")

    return comparison


def create_visualizations(comparison: pd.DataFrame) -> None:
    """Create comparison visualizations."""
    output_dir = Path("experiments/results")

    # Set style
    sns.set_style("whitegrid")
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # 1. Test RMSE Comparison
    ax = axes[0, 0]
    x = range(len(comparison))
    width = 0.35
    ax.bar(
        [i - width / 2 for i in x],
        comparison["OLD Test RMSE"],
        width,
        label="OLD (weekday+rush_hour)",
        alpha=0.8,
        color="#3498db",
    )
    ax.bar(
        [i + width / 2 for i in x],
        comparison["NEW Test RMSE"],
        width,
        label="NEW (hour_of_day+day_of_week)",
        alpha=0.8,
        color="#e74c3c",
    )
    ax.set_xlabel("Model Config", fontsize=11)
    ax.set_ylabel("Test RMSE (minutes)", fontsize=11)
    ax.set_title("Test RMSE: Old vs New Transformers", fontsize=12, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(comparison["Config"])
    ax.legend()
    ax.grid(axis="y", alpha=0.3)

    # 2. Test R² Comparison
    ax = axes[0, 1]
    ax.bar(
        [i - width / 2 for i in x],
        comparison["OLD Test R²"],
        width,
        label="OLD",
        alpha=0.8,
        color="#3498db",
    )
    ax.bar(
        [i + width / 2 for i in x],
        comparison["NEW Test R²"],
        width,
        label="NEW",
        alpha=0.8,
        color="#e74c3c",
    )
    ax.set_xlabel("Model Config", fontsize=11)
    ax.set_ylabel("Test R² Score", fontsize=11)
    ax.set_title("Test R²: Old vs New Transformers", fontsize=12, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(comparison["Config"])
    ax.legend()
    ax.set_ylim([0.7, 0.85])
    ax.grid(axis="y", alpha=0.3)

    # 3. RMSE Difference
    ax = axes[1, 0]
    colors = ["green" if d < 0 else "red" for d in comparison["RMSE Diff %"]]
    ax.barh(comparison["Config"], comparison["RMSE Diff %"], color=colors, alpha=0.7)
    ax.set_xlabel("RMSE Change (%)", fontsize=11)
    ax.set_title(
        "Performance Change: New vs Old\n(Negative = Improvement)",
        fontsize=12,
        fontweight="bold",
    )
    ax.axvline(x=0, color="black", linestyle="--", linewidth=1)
    ax.grid(axis="x", alpha=0.3)

    # 4. Training Time Comparison
    ax = axes[1, 1]
    ax.bar(
        [i - width / 2 for i in x],
        comparison["OLD Train Time (s)"],
        width,
        label="OLD",
        alpha=0.8,
        color="#3498db",
    )
    ax.bar(
        [i + width / 2 for i in x],
        comparison["NEW Train Time (s)"],
        width,
        label="NEW",
        alpha=0.8,
        color="#e74c3c",
    )
    ax.set_xlabel("Model Config", fontsize=11)
    ax.set_ylabel("Training Time (seconds)", fontsize=11)
    ax.set_title(
        "Training Time: Old vs New Transformers", fontsize=12, fontweight="bold"
    )
    ax.set_xticks(x)
    ax.set_xticklabels(comparison["Config"])
    ax.legend()
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    viz_file = output_dir / "transformer_comparison.png"
    plt.savefig(viz_file, dpi=300, bbox_inches="tight")
    print(f"✅ Saved visualization: {viz_file}")
    plt.close()


def main() -> None:
    print("=" * 80)
    print("TRANSFORMER COMPARISON ANALYSIS")
    print("=" * 80)

    # Save results
    df = save_results()

    # Create comparison
    comparison = create_comparison_table(df)

    print("\n" + "=" * 80)
    print("COMPARISON SUMMARY")
    print("=" * 80)
    print(comparison.to_string(index=False))

    # Create visualizations
    create_visualizations(comparison)

    print("\n" + "=" * 80)
    print("FILES SAVED")
    print("=" * 80)
    print("📁 experiments/results/")
    print("  ├── transformer_comparison_results.csv")
    print("  ├── transformer_comparison_results.xlsx")
    print("  ├── comparison_summary.csv")
    print("  └── transformer_comparison.png")

    print("\n✅ All files saved successfully!")


if __name__ == "__main__":
    main()
