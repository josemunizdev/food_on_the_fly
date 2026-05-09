# Contributing to Food on the Fly

This is the team onboarding doc. If you can clone the repo and follow this guide top to bottom, you should be running training locally inside an hour.

## Prerequisites

You'll need the following installed on your machine before starting:

| Tool | What it's for | Install |
|---|---|---|
| Homebrew | Package manager (Mac) | https://brew.sh |
| Python 3.13 | Project runtime (matches CI) | `brew install python@3.13` |
| Git | Source control | Comes with Xcode CLI tools, or `brew install git` |
| gcloud CLI | GCP auth and bucket access | `brew install --cask google-cloud-sdk` |

Verify each one:

```bash
brew --version
python3.13 --version
git --version
gcloud --version
```

If any of those error, install before continuing.

## One-time setup

### 1. Clone the repo

```bash
git clone https://github.com/josemunizdev/food_on_the_fly.git
cd food_on_the_fly
```

### 2. Create the project virtual environment

The project is pinned to Python 3.13 (matches CI). Always use a venv, never install project deps into your system Python.

```bash
python3.13 -m venv .venv
source .venv/bin/activate
```

Your prompt should now show `(.venv)` at the start. If it doesn't, the activation didn't work and the rest will fail.

### 3. Install the project and dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
pip install -r requirements_dev.txt
pip install -e .
pre-commit install
```

The `pip install -e .` puts the project itself on your path so `from food_on_the_fly import ...` works in scripts. `pre-commit install` wires up the formatter and linter to run automatically on `git commit`.

You can also run all of that with `make dev` once the venv is active.

### 4. Fix Python SSL certificates (python.org installer only)

If you installed Python from python.org instead of Homebrew, run this once or you'll hit `SSL: CERTIFICATE_VERIFY_FAILED` errors when reading from GCS:

```bash
/Applications/Python\ 3.13/Install\ Certificates.command
```

Homebrew Python users can skip this step.

### 5. Authenticate with GCP

The dataset and model artifacts live in a private GCS bucket. You need two auth steps. Both happen on your local machine, not in Cloud Shell.

```bash
# Auth the gcloud CLI itself
gcloud auth login

# Auth Application Default Credentials (used by pandas, gcsfs, the Python SDKs)
gcloud auth application-default login

# Point gcloud at our project
gcloud config set project project-9aed1f8e-f40e-4a15-858
```

Each command opens a browser tab. Authorize with the Google account that has been added to the project, return to terminal.

If you haven't been added to the GCP project or the GCS bucket yet, ask Jose to grant you `Storage Object Viewer` on the bucket. Without that, the next step will return a 403.

### 6. Verify GCS read access

```bash
gcloud storage ls gs://project-9aed1f8e-f40e-4a15-858-models/data/raw/
python scripts/test_data_load.py
```

The first command should list the dataset CSV. The second should print the dataset shape (~45k rows) and column names.

If either fails, see Troubleshooting at the bottom of this doc.

## Daily workflow

### Branch and PR rules

These are enforced by the `PR checks` GitHub Actions workflow. PRs that violate them are blocked at the check stage, not at merge.

- **Feature work goes on `feature/<name>` branches**, fixes go on `fix/<name>` branches. No other prefixes are allowed.
- **PRs into `dev` must come from a `feature/*` or `fix/*` branch.** Never PR directly to main.
- **PRs into `main` must come from `dev`.** Use main only for releases.
- **All PRs are reviewed by GitHub Copilot automatically** plus at least one human teammate.

The flow:

```
feature/add-distance-features → dev → main
fix/null-traffic-density       → dev → main
```

### Cutting a branch and opening a PR

```bash
# Always start from latest dev
git checkout dev
git pull

# Cut a feature or fix branch
git checkout -b feature/add-distance-features

# Make your changes...
# Run lint and tests locally before pushing:
make lint
make test

# Commit, push
git add .
git commit -m "Add haversine distance feature"
git push -u origin feature/add-distance-features
```

Then open a PR on GitHub targeting `dev` (not main).

### Useful `make` targets

| Command | What it does |
|---|---|
| `make dev` | Install everything (deps, project, pre-commit hooks) |
| `make test` | Run pytest |
| `make lint` | Run ruff lint and format check |
| `make format` | Auto-fix lint and format issues |
| `make train` | Run training (once `train_model.py` is implemented) |
| `make data` | Reserved for data prep tasks |
| `make docker_build` | Build the Cloud Run container locally |
| `make clean` | Remove caches and build artifacts |

## How the data flows

You don't manually download datasets. Here's why:

```
Kaggle (raw source)
  │
  │ Jose downloads once via kagglehub
  ▼
gs://project-9aed1f8e-f40e-4a15-858-models/data/raw/deliverytime.csv  ← single source of truth
  │
  ▼
Your laptop reads directly from GCS via pandas + gcsfs
  │
  ▼
data/processed/   ← local cache for processed splits, gitignored
  │
  ▼
models/  ← trained models written here, then uploaded to GCS
```

Reading from GCS works in three places using the exact same code path:
- Your laptop (with ADC from Step 5 above)
- GitHub Actions (with WIF, no setup needed by you)
- Vertex AI / Cloud Run (with the runtime service account, no setup)

So the same training script runs identically locally and in cloud.

## How CI/CD works

Two GitHub Actions workflows govern the repo:

### `pr-checks.yml` runs on every PR

Three sequential jobs. Each must pass before the next runs:

1. **Validate branch name** — must be `feature/*` or `fix/*` for PRs to dev.
2. **Validate branch direction** — must be `feature/* | fix/* → dev` or `dev → main`.
3. **Lint and test** — ruff, mypy, pytest.

If any job fails, downstream jobs are skipped and the PR is blocked.

### `deploy.yml` runs after PR-checks succeeds on `main`

Triggered via `workflow_run` chaining. Sequence:

1. PR is merged to main.
2. `pr-checks` re-runs against the merge commit.
3. If it passes, `deploy` fires automatically.
4. Deploy authenticates to GCP via Workload Identity Federation, builds the Docker image, pushes to Artifact Registry, and deploys to Cloud Run.

You don't need any GCP credentials on your machine for deploy to work. WIF handles auth for GitHub Actions automatically.

## Troubleshooting

**`SSL: CERTIFICATE_VERIFY_FAILED` when reading from GCS** — Run the Install Certificates script in Step 4. python.org installer skips this by default.

**`ModuleNotFoundError: No module named 'food_on_the_fly'`** — Either your venv isn't active (`source .venv/bin/activate`), or you didn't run `pip install -e .`.

**`ModuleNotFoundError: No module named 'gcsfs'`** — `pip install -r requirements.txt` to refresh deps.

**`403 Forbidden` reading from GCS** — You haven't been added to the bucket. Ping Jose to grant you `Storage Object Viewer`.

**`401` or `Reauthentication required`** — ADC token expired. Run `gcloud auth application-default login` again.

**Pasting multi-line commands hangs in Terminal** — Probably an unclosed quote. Hit Ctrl+C, try again. Better: save to a file and run with `python <file>`.

**Tests fail locally but pass on someone else's machine** — Check your Python version. Run `python --version` inside the venv. Should be 3.13.x. If it's not, recreate the venv with `python3.13 -m venv .venv`.

**`pre-commit` complaining on every commit** — That's it doing its job. Run `make format` to auto-fix.

## Want Claude in your editor?

The project is set up to play well with Claude Code in VS Code. Install the Claude Code VS Code extension, open the repo, and Claude will read the conventions from any `CLAUDE.md` we drop in the project root. Useful for keeping every teammate's AI assistant aligned on project rules.

Not required for any of the above to work. Just a nice-to-have.

## Need help

If you're stuck on something not covered here, check the Troubleshooting section first, then ask in our group chat or open a GitHub issue. If you find a gotcha worth documenting, please update this file as part of your next PR.
