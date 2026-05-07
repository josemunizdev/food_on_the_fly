# PHASE 1: Project Design & Model Development

## Overview

Phase 1 establishes the foundation for your MLOps project. This phase covers project planning, initial code organization, team collaboration setup, data handling, baseline model development, and comprehensive documentation. By the end of this phase, you should have a well-organized repository with a trained baseline model and clear documentation for future team members.

---

## 1. Project Proposal

- [x] **Scope & Objectives**: Define the problem statement, goals, and success metrics for Food on the Fly
- [x] **Detailed Description**: Write a 300+ word project description covering the business context, technical approach, and expected outcomes
- [x] **Dataset Selection**: Zomato Delivery Operations Analytics dataset (Kaggle) — ~45 000 delivery records with GPS, weather, traffic, and vehicle features
- [x] **Dataset Description**: 19 features including lat/lon pairs, order timestamps, weather conditions, road traffic density, vehicle type, and target `Time_taken (min)`
- [x] **Model Considerations**: Regression task; baseline with sklearn pipeline; HaversineTransformer for distance feature; candidates include GradientBoosting, RandomForest, XGBoost
- [x] **Open-Source Tools**: scikit-learn (modeling), DVC (data versioning), MLflow (experiment tracking), Hydra (config), ruff/mypy (code quality), pytest (testing), GitHub Actions (CI/CD)

---

## 2. Code Organization & Setup

- [x] **GitHub Repository**: Created with SE 489 MLOps cookiecutter structure
- [x] **Environment Setup**: `pyproject.toml` defines Python 3.13+ requirement; `pip install -e .` for editable install
- [x] **Dependency Management**: `requirements.txt` (runtime) and `requirements_dev.txt` (dev tools) maintained
- [x] **Project Structure**: `src/` layout with clear separation — `data/`, `features/`, `models/`, `evaluation/`, `visualization/`, `utils/`
- [x] **Version Pinning**: All dependencies pinned with `>=` lower bounds in `pyproject.toml` and `requirements.txt`
- [x] **Installation Documentation**: README covers pip and uv install paths, editable install, and dev setup

---

## 3. Version Control & Collaboration

- [x] **Regular Commits**: Atomic, descriptive commits following conventional commit style (`feat:`, `fix:`, `docs:`)
- [x] **Branching Strategy**: `feature/* → dev → main` workflow enforced via PR template and CI
- [x] **Pull Request Process**: PR template in `.github/PULL_REQUEST_TEMPLATE.md`; CI must pass before merge
- [x] **Team Roles**: Project Lead (Jose), team members Abdul, Aviv, Imran defined in README
- [ ] **Code Review Guidelines**: To be documented in `CONTRIBUTING.md`
- [x] **Commit History**: Clean history; bad merges reverted and re-routed through dev

---

## 4. Data Handling

- [x] **Data Cleaning Scripts**: `make_dataset.py` — loads Zomato CSV, cleans target column, filters bad rows, parses dates, produces temporal + random splits
- [ ] **Normalization**: To be implemented in sklearn pipeline during model training
- [ ] **Data Augmentation**: Not applicable for tabular regression task
- [x] **Data Documentation**: 19 features documented — lat/lon pairs, timestamps, weather, traffic density, vehicle type, multiple deliveries, festival flag, city tier
- [x] **Data Splits**: 70 / 15 / 15 train/val/test on first 40 days; last 14 days reserved as drift set
- [ ] **Data Validation**: To be implemented
- [x] **DVC Setup**: DVC initialized; `data/processed/` (train, val, test, drift CSVs) tracked with `.dvc` files

---

## 5. Model Training

- [ ] **Training Environment**: Local CPU; GPU not required for tabular regression
- [x] **Baseline Model**: `BaseModel` ABC + `Model` scaffold in `src/food_on_the_fly/models/`; training entrypoint in `train_model.py`
- [ ] **Hyperparameter Configuration**: `configs/config.yaml` via Hydra — to be populated with chosen estimator params
- [x] **Evaluation Metrics**: `evaluation/metrics.py` — `regression_report` (MAE, MSE, RMSE, R²) and `classification_report`
- [x] **Model Persistence**: `Model.save()` / `Model.load()` via joblib
- [x] **Training Reproducibility**: `utils/seed.py` seeds Python, NumPy, and optional torch/tf RNGs; `--seed` CLI arg
- [ ] **Performance Baseline**: To be documented after first training run

---

## 6. Documentation & Reporting

- [x] **README**: Comprehensive README with:
  - [x] Project overview and objectives
  - [x] Setup and installation instructions
  - [x] Quick start guide for running training
  - [x] Dependencies and requirements
  - [ ] Contributing guidelines (CONTRIBUTING.md pending)
  - [x] License information
- [x] **Code Docstrings**: Module-level and function docstrings throughout `src/`
- [x] **Code Style**: ruff configured in `pyproject.toml` (`E`, `F`, `I`, `N`, `W`, `B`, `UP` rules, 88-char line length)
- [x] **Type Hints**: Type annotations throughout codebase
- [x] **Type Checking**: mypy configured (`disallow_untyped_defs = true`, `warn_return_any = true`)
- [x] **Makefile**: Commands for `install`, `dev`, `data`, `train`, `predict`, `test`, `lint`, `format`, `clean`, `docker_build`, `docker_run`, `docs`
- [ ] **CONTRIBUTING.md**: To be created
- [ ] **API Documentation**: Planned for Phase 2/3 (FastAPI)

---

> **Checklist:** Use this as a guide for documenting your Phase 1 deliverables.
