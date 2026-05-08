# Phase 1 Report – Food on the Fly

## Phase 1: Project Design & Model Development

### Overview

Phase 1 establishes the foundation for the MLOps project. This phase covers project planning, initial code organization, team collaboration setup, data handling, baseline model development, and comprehensive documentation. By the end of this phase, we have a well-organized repository with a trained baseline model and clear documentation for future team members.

---

### 1. Project Proposal

- [x] **Scope & Objectives**: Define the problem statement, goals, and success metrics for Food on the Fly
- [x] **Detailed Description**: Write a 300+ word project description covering the business context, technical approach, and expected outcomes
- [x] **Dataset Selection**: Zomato Delivery Operations Analytics dataset (Kaggle) — ~45 000 delivery records with GPS, weather, traffic, and vehicle features
- [x] **Dataset Description**: 19 features including lat/lon pairs, order timestamps, weather conditions, road traffic density, vehicle type, and target `Time_taken (min)`
- [x] **Model Considerations**: Regression task; baseline with sklearn pipeline; HaversineTransformer for distance feature; candidates include GradientBoosting, RandomForest, XGBoost
- [x] **Open-Source Tools**: scikit-learn (modeling), DVC (data versioning), MLflow (experiment tracking), Hydra (config), ruff/mypy (code quality), pytest (testing), GitHub Actions (CI/CD)

---

### 2. Code Organization & Setup

- [x] **GitHub Repository**: Created with SE 489 MLOps cookiecutter structure
- [x] **Environment Setup**: `pyproject.toml` defines Python 3.13+ requirement; `pip install -e .` for editable install
- [x] **Dependency Management**: `requirements.txt` (runtime) and `requirements_dev.txt` (dev tools) maintained
- [x] **Project Structure**: `src/` layout with clear separation — `data/`, `features/`, `models/`, `evaluation/`, `visualization/`, `utils/`
- [x] **Version Pinning**: All dependencies pinned with `>=` lower bounds in `pyproject.toml` and `requirements.txt`
- [x] **Installation Documentation**: README covers pip and uv install paths, editable install, and dev setup

---

### 3. Version Control & Collaboration

- [x] **Regular Commits**: Atomic, descriptive commits following conventional commit style (`feat:`, `fix:`, `docs:`)
- [x] **Branching Strategy**: `feature/* → dev → main` workflow enforced via PR template and CI
- [x] **Pull Request Process**: PR template in `.github/PULL_REQUEST_TEMPLATE.md`; CI must pass before merge
- [x] **Team Roles**: Project Lead (Jose), team members Abdul, Aviv, Imran defined in README
- [ ] **Code Review Guidelines**: To be documented in `CONTRIBUTING.md`
- [x] **Commit History**: Clean history; bad merges reverted and re-routed through dev

---

### 4. Data Handling

- [x] **Data Cleaning Scripts**: `make_dataset.py` — loads Zomato CSV, cleans target column, filters bad rows, parses dates, produces temporal + random splits
- [ ] **Normalization**: To be implemented in sklearn pipeline during model training
- [ ] **Data Augmentation**: Not applicable for tabular regression task
- [x] **Data Documentation**: 19 features documented — lat/lon pairs, timestamps, weather, traffic density, vehicle type, multiple deliveries, festival flag, city tier
- [x] **Data Splits**: 70 / 15 / 15 train/val/test on first 40 days; last 14 days reserved as drift set
- [ ] **Data Validation**: To be implemented
- [x] **DVC Setup**: DVC initialized; `data/processed/` (train, val, test, drift CSVs) tracked with `.dvc` files

---

### 5. Model Training

- [x] **Training Environment**: Local CPU; GPU not required for tabular regression
- [x] **Baseline Model**: XGBoost regressor via sklearn Pipeline; training entrypoint in `train_model.py` with Hydra config and MLflow tracking
- [x] **Hyperparameter Configuration**: `configs/config.yaml` via Hydra — 100 estimators, max depth 6, learning rate 0.1; all overridable from CLI
- [x] **Evaluation Metrics**: `evaluation/metrics.py` — `regression_report` (MAE, MSE, RMSE, R²) and `classification_report`
- [x] **Model Persistence**: Pipeline serialized with `mlflow.sklearn.log_model`; also saved locally via joblib
- [x] **Training Reproducibility**: `utils/seed.py` seeds Python, NumPy, and optional torch/tf RNGs; random state pinned at 42
- [x] **Performance Baseline**: Val RMSE 4.01 min, MAE 3.19 min, R² 0.8207; train R² 0.8354 (minimal overfitting)

---

### 6. Documentation & Reporting

- [x] **README**: Comprehensive README with project overview, setup instructions, quick start, dependencies, and license
- [x] **Code Docstrings**: Module-level and function docstrings throughout `src/`
- [x] **Code Style**: ruff configured in `pyproject.toml` (`E`, `F`, `I`, `N`, `W`, `B`, `UP` rules, 88-char line length)
- [x] **Type Hints**: Type annotations throughout codebase
- [x] **Type Checking**: mypy configured (`disallow_untyped_defs = true`, `warn_return_any = true`)
- [x] **Makefile**: Commands for `install`, `dev`, `data`, `train`, `predict`, `test`, `lint`, `format`, `clean`, `docker_build`, `docker_run`, `docs`
- [ ] **CONTRIBUTING.md**: To be created
- [ ] **API Documentation**: Planned for Phase 2/3 (FastAPI)

---

## Model Training and Resources

The training pipeline for phase one came down to three tools: Hydra for config, MLflow for tracking experiments, and XGBoost as the actual model. That's it. We wanted something where we could change hyperparameters from the command line and have the whole run logged automatically, and this stack does that.

Preprocessing is a sklearn Pipeline with three transformers running in parallel through a ColumnTransformer. The first one is a HaversineTransformer, which calculates the great-circle distance between the restaurant and the delivery coordinates. The parameters come in from a YAML config, which Hydra manages. Then a StandardScaler handles the numeric features: delivery person age, ratings, vehicle condition, and multiple deliveries count. OneHotEncoder takes care of the categoricals: weather, traffic density, order type, vehicle type, festival flag, city. This is basic preprocessing that doesn't break when the data changes at this point.

The XGBoost regressor runs 100 estimators, max depth of 6, learning rate of 0.1. All of that is configurable through Hydra, so if we want to try 200 estimators we can just pass it on the command line instead of editing a file. Every run logs to MLflow: hyperparams, all the metrics (MSE, RMSE, MAE, R²), and the serialized model. We also have mypy running on the codebase, which required some explicit isinstance checks around the OmegaConf containers because the static analysis doesn't like their types otherwise, we were getting numerous problems running the code.

The data is pre-split through DVC. There are 21,513 training samples and 4,610 validations. Training finishes in about one to two seconds on multi-core CPU because XGBoost runs with n_jobs=-1. The baseline lands at RMSE of 4.01 minutes, MAE of 3.19, R² of 0.8207. So we're explaining about 82% of the variance in delivery times. Training R² is 0.8354, which means there's minimal overfitting and the gap between training and validation is small enough that we are happy about it for a first run.

For the tooling side: XGBoost 2.0+, MLflow 2.16+, Hydra 1.3+, sklearn 1.5+. Ruff and black run through pre-commit hooks so nothing gets committed with formatting issues. Reproducibility was a priority, so random seeds pinned at 42, package versions locked in requirements.txt, uv handling resolution. MLflow logs everything we could think of: model signatures, dependency specs, git commit hashes. Each experiment takes about 100MB of MLflow artifacts, though the sklearn pipeline itself is only around 5MB. Keeps iteration fast when you're not waiting on storage.

---

## Challenges and What's Next

The integration work was harder than we expected. Getting Google OAuth working was the first roadblock — took a few rounds of debugging the credentials and redirect URIs before it actually cooperated. Not a deep technical problem, just the kind of thing where you're staring at config files wondering what's wrong until you find it.

The real pain was mypy. We wanted type safety across the whole codebase, which sounds reasonable until you try to pass OmegaConf containers to MLflow. OmegaConf.to_container() returns this union type — dict[str | bytes | int | Enum | float | bool, Any] | dict[Any, Any] — and MLflow wants dict[str, Any]. Those aren't compatible. So we ended up writing isinstance checks and dictionary comprehensions to force all keys to strings before MLflow would accept them. It's defensive programming that shouldn't be necessary, but here we are.

Then there's the linter situation. Ruff for linting, black for formatting, mypy for types — getting all three to agree on what "correct" looks like took more effort than we'd admit. Import ordering, 88-character line limits, and the classic ML problem where convention says you name your variables X and y but the linter thinks you're using the wrong case. We ended up adding a selective rule ignore for N806 in pyproject.toml so it would stop yelling about uppercase variable names. Standard ML notation and strict naming rules don't really get along.

On the infrastructure side, Docker is still missing. That's a real gap — without containerization, the pipeline isn't reproducible outside of whoever's machine it was built on. We know it needs to happen, it just hasn't happened yet.

The Git workflow was its own learning curve. Feature branches, merge conflicts, branch protection rules required two approving reviews and passing CI before anything could hit main, which is good practice but adds friction when the team is still figuring out the process together. The pull request cycle — code reviews, automated linter feedback, keeping commit history clean — made it obvious that the pre-commit hooks needed to be stricter earlier. Would've saved everyone time.

One thing that took real thought was how DVC, Git, Hydra, and MLflow all coexist. Each tool generates its own artifacts (.dvc files, mlruns/, outputs/) and you have to be careful about what gets gitignored and what gets tracked. Get it wrong and you're either committing massive binary files or losing your experiment history. The configuration files are what tie it all together, so those stay tracked.

For what's next: hyperparameter tuning will be the focus and ensuring reproducibility. The current model uses defaults basically: 100 estimators, max depth 6, learning rate 0.1. Running GridSearchCV or Optuna to actually search the space should improve things. We also want to try other architectures, and maybe throw a neural network at it just to see where the ceiling is for delivery time prediction.

The thing we keep coming back to is that we don't have a clear definition of "good enough." The model hits 4.01-minute RMSE, but is that acceptable? Does the business care about being within 5 minutes? 10% relative error? That threshold changes everything about how much effort the tuning is worth and what tradeoffs make sense. The model complexity versus accuracy, customer expectations versus what's actually achievable given the data, how specific the driver scheduling needs to be.

Beyond that, Docker needs to happen, CI/CD should include automated hyperparameter sweeps, and there should be some kind of monitoring dashboard watching for data drift on the holdout set. Right now this is an experimental prototype. Getting it to a place where it retrains automatically when performance drops and perhaps to do a shadow deployment are the goals, but there's real work between here and there.
