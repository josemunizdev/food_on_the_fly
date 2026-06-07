# Continuous Machine Learning (CML)

This project uses [CML](https://cml.dev/) to automatically train the model and
report results on every pull request. When a PR is opened against `main` or
`dev`, a GitHub Actions runner trains the model, generates performance metrics
and plots, compares the result against a committed baseline, and posts all of it
back as a comment on the PR.

The workflow is defined in `.github/workflows/cml.yml`.

## What the workflow does

On each run the job:

1. Trains an XGBoost model on the GitHub-hosted runner using the
   `xgboost_baseline` configuration.
2. Writes validation/training metrics to `metrics.txt`.
3. Generates four diagnostic plots (prediction scatter, residuals, error
   distribution, feature importance).
4. Compares the freshly trained model against the committed baseline in
   `reports/baseline_metrics.json`.
5. Posts a single Markdown comment on the PR containing the metrics, the
   comparison table, and the plots.

## Triggers

The workflow runs on:

- `push` to `main` or `dev`
- `pull_request` targeting `main` or `dev`

The PR comment (step 8) is only created on `pull_request` events. Pushes train
the model and validate the pipeline but do not post a comment.

## Prerequisites

The workflow reads data splits from Google Cloud Storage, so it authenticates to
GCP using Workload Identity Federation. Two repository secrets must be
configured under **Settings -> Secrets and variables -> Actions**:

| Secret | Purpose |
|---|---|
| `GCP_WIF_PROVIDER` | Workload Identity Federation provider for keyless auth |
| `GCP_SERVICE_ACCOUNT` | Service account used to read the GCS data bucket |

`GITHUB_TOKEN` is provided automatically by GitHub Actions and is used by CML to
post the PR comment. No manual setup is required for it, but the workflow must
declare `pull-requests: write` permission (it does).

## Pipeline steps

The job (`train-and-report`, `runs-on: ubuntu-latest`) runs these steps in
order:

1. **Checkout** the repository.
2. **Install uv** and **set up Python 3.11** (uv is used instead of pip for
   speed).
3. **Setup CML** via `iterative/setup-cml`.
4. **Install dependencies** into a uv virtual environment (`uv pip install -e .`).
5. **Authenticate to Google Cloud** using the WIF secrets above.
6. **Process data** by running `food_on_the_fly.data.temporal_split` to generate
   the train/validation/test splits.
7. **Train the model** with `python -m food_on_the_fly.train_model
   model=xgboost_baseline`, which writes `metrics.txt` and the plot PNGs.
8. **Create and post the CML report** (PRs only).

## Metrics format

`train_model.py` writes `metrics.txt` in a plain-text, two-section format:
=== TRAINING METRICS ===
RMSE: 5.08 minutes
MAE:  3.99 minutes
R2:   0.7088
=== VALIDATION METRICS ===
RMSE: 5.08 minutes
MAE:  3.99 minutes
R2:   0.7088The CML comment embeds this verbatim inside a code block. RMSE and MAE are in
minutes; R2 is the coefficient of determination.

## Plots

Four PNGs are generated during training and embedded in the comment:

| Plot | File | Shows |
|---|---|---|
| Prediction vs Actual | `prediction_scatter.png` | How closely predictions track actual delivery times |
| Residual Plot | `residual_plot.png` | Whether prediction errors are randomly distributed |
| Error Distribution | `error_distribution.png` | Distribution of prediction errors (minutes) |
| Feature Importance | `feature_importance.png` | Which features drive the prediction |

These files are regenerated on every run and should not be committed to version
control.

## Model comparison against baseline

Step 8 runs `scripts/compare_to_baseline.py`, which:

1. Parses the validation metrics from `metrics.txt`.
2. Reads the reference metrics from `reports/baseline_metrics.json`.
3. Writes a Markdown comparison table to `comparison.md`, which is appended to
   the PR comment.

The table reports the delta for each metric and flags whether the current model
is `better`, `worse`, or the `same` as the baseline (lower is better for
RMSE/MAE, higher is better for R2).

`reports/baseline_metrics.json` looks like:

```json
{
  "model": "xgboost_baseline",
  "description": "CML reference baseline ...",
  "val_rmse": 5.08,
  "val_mae": 3.99,
  "val_r2": 0.7088
}
```

The baseline tracks the `xgboost_baseline` config so that comparisons are
like-for-like with the model the CML workflow trains. Note that `REPORT.md`
documents a larger configuration (100 trees, depth 6) with stronger numbers;
the baseline file intentionally tracks the smaller CML config instead.

## Customization

**Change the model the workflow trains.** Edit the train step in `cml.yml` and
override the Hydra config group, e.g. `model=xgboost_optimized`:

```yaml
python -m food_on_the_fly.train_model model=xgboost_optimized
```

If you change the trained config, update `reports/baseline_metrics.json` to the
new config's numbers so the comparison stays like-for-like (see below).

**Update the baseline.** When the reference model legitimately changes, run the
new model, read its validation RMSE/MAE/R2 from `metrics.txt`, and edit the
three numeric fields in `reports/baseline_metrics.json`. Commit the change with
a message explaining why the baseline moved.

**Test the comparison locally** without running the full workflow:

```bash
python scripts/compare_to_baseline.py --metrics metrics.txt
# or against a saved metrics file:
python scripts/compare_to_baseline.py --metrics path/to/metrics.txt \
    --baseline reports/baseline_metrics.json --out comparison.md
```

**Add or change plots.** Plots are produced inside `train_model.py`. Add the new
PNG there, then add a corresponding `echo '![...](./your_plot.png)'` block in
step 8 of `cml.yml` so it appears in the comment.

**Change triggers.** Edit the `on:` block at the top of `cml.yml`. To run on
additional branches, extend the `branches` lists.

## Troubleshooting

- **No comment appears on the PR.** Confirm the event is a `pull_request` (the
  comment step is skipped on plain pushes) and that the workflow has
  `pull-requests: write` permission.
- **Auth/data step fails.** Verify `GCP_WIF_PROVIDER` and `GCP_SERVICE_ACCOUNT`
  are set and that the service account can read the data bucket.
- **Comparison shows `worse` on an unchanged model.** The baseline in
  `reports/baseline_metrics.json` was likely captured from a different config or
  data split than the workflow trains. Recapture it from the current
  `xgboost_baseline` run.
