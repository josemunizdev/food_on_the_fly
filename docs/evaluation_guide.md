# Model Evaluation Guide

# Model selection mode: best 5/29/2026

Finding best model by metric: val_rmse (min)
 Found 65 runs with val_rmse
 Best model: 41a703028bec49e09ca307123110d66a
 val_rmse: 3.9625

Test Results:
 RMSE: 3.98 minutes
 MAE: 2.95 minutes
 R²: 0.8456
 Degradation: +0.5% (excellent generalization)

## Overview

This guide explains how to use the model evaluation system
 to test trained models on the test set.

## When to Evaluate

**Important**: Only evaluate on the test set **after**
 you've completed hyperparameter tuning!

### Workflow

1. **Training Phase**: Train multiple models with different
   hyperparameters
2. **Selection Phase**: Compare validation metrics in MLflow
   UI
3. **Evaluation Phase**: Evaluate the best model (once!) on
   test set
4. **Decision Phase**: Deploy if test performance is
   acceptable

## Evaluation Modes

### 1. Best Mode (Recommended)

Automatically finds and evaluates the model with the best
 validation metric.

```bash
make evaluate_best

Use when:
- Final evaluation before deployment
- After hyperparameter tuning
- You want to ensure you're testing the best model

2. Latest Mode

Evaluates the most recently trained model.

make evaluate_latest

Use when:
- Quick check after training
- Development testing
- You just trained a model and want to see test performance

3. Specific Mode

Evaluates a specific model by run ID.

make evaluate evaluation.selection_mode=specific
evaluation.run_id=abc123def456

Use when:
- Re-evaluating a specific model
- Testing a model from a specific experiment
- Reproducing previous results

What Gets Tracked

Each evaluation run logs to MLflow:

Parameters

- parent_run_id: Training run ID
- parent_run_name: Training run name
- model_*: All hyperparameters from training (e.g.,
model_n_estimators)
- evaluation_type: "baseline_test"
- test_samples: Number of test samples

Metrics

- test_rmse, test_mae, test_r2: Test set performance
- parent_val_rmse, parent_val_mae, parent_val_r2: Validation
 performance
- rmse_degradation: Difference between test and validation
RMSE
- rmse_degradation_pct: Percentage degradation
- eval_duration_seconds: Evaluation time

Interpreting Results

Degradation Metrics

The rmse_degradation_pct tells you how well the model
generalizes:
┌────────┬───────────────────┬─────────────────────────────┐
│ Value  │      Meaning      │           Action            │
├────────┼───────────────────┼─────────────────────────────┤
│ ~0%    │ Perfect           │ ✅ Deploy with confidence   │
│        │ generalization    │                             │
├────────┼───────────────────┼─────────────────────────────┤
│ +5% to │ Slight            │ ⚠️ Acceptable, monitor in   │
│  +10%  │ degradation       │ production                  │
├────────┼───────────────────┼─────────────────────────────┤
│        │ Significant       │ ❌ Overfitting! Retrain     │
│ > +10% │ degradation       │ with more                   │
│        │                   │ data/regularization         │
├────────┼───────────────────┼─────────────────────────────┤
│ < -10% │ Test better than  │ 🤔 Suspicious! Check for    │
│        │ validation        │ data leakage                │
└────────┴───────────────────┴─────────────────────────────┘
Example: Good Generalization

Validation RMSE: 3.96 minutes
Test RMSE: 3.98 minutes
Degradation: +0.02 minutes (+0.5%)
✅ Test performance matches validation (good generalization)

Example: Overfitting Detected

Validation RMSE: 3.96 minutes
Test RMSE: 4.45 minutes
Degradation: +0.49 minutes (+12.4%)
⚠️ Test performance differs from validation by >10%!

Configuration

Edit configs/config.yaml:

evaluation:
  selection_mode: "latest"  # Change to "best" for
production
  best_metric: "val_rmse"   # Metric to optimize
  best_metric_order: "min"  # min for RMSE/MAE, max for R²
  run_id: null              # Set for specific mode

Common Workflows

Workflow 1: Hyperparameter Tuning

# Train variants
make train model=xgboost_baseline
make train model=xgboost_optimized

# Check validation metrics
mlflow ui

# Evaluate best model on test set
make evaluate_best

Workflow 2: A/B Testing

# Evaluate model A
make evaluate evaluation.selection_mode=specific
evaluation.run_id=modelA_id

# Evaluate model B
make evaluate evaluation.selection_mode=specific
evaluation.run_id=modelB_id

# Compare in MLflow UI

Workflow 3: Production Deployment Decision

# After training many models...
make evaluate_best

# Check:
# 1. Is test RMSE acceptable for business requirements?
# 2. Is degradation < 10%?
# 3. Are all hyperparameters logged for reproducibility?

# If yes → Deploy!
# If no → Continue tuning

Troubleshooting

"No runs found with metric val_rmse"

Cause: No training runs have validation metrics logged.

Solution: Train a model first:
make train

"Model not found in evaluation run"

Cause: Accidentally trying to load a model from an
evaluation run.

Solution: Use "best" or "latest" mode, which filters out
evaluation runs automatically.

"Run does not exist"

Cause: Invalid run_id in specific mode.

Solution: Check the run ID in MLflow UI and copy the correct
 ID.

Best Practices

1. Evaluate sparingly: Only evaluate on test set 1-2 times
per project
2. Use validation for tuning: Never tune hyperparameters
based on test performance
3. Document decisions: Note why you selected a particular
model
4. Track everything: The system logs all hyperparameters
automatically
5. Check generalization: Always review degradation metrics
before deployment
```
