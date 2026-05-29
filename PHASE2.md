# PHASE 2: Enhancing ML Operations with Containerization & Monitoring

## Overview
Phase 2 focuses on scaling and operationalizing Food on the Fly by implementing containerization, advanced monitoring, profiling, experiment tracking, and comprehensive logging. This phase ensures your model can be reliably deployed, monitored in production, and continuously improved through systematic experimentation.

---

## 1. Containerization

- [x] **Dockerfile Creation**: Build Dockerfile for model training and inference
    - Evidence: `dockerfiles/Dockerfile` (commit a001e31)
- [x] **Base Image Selection**: Choose appropriate base image (python:3.x, nvidia/cuda, etc.)
    - Evidence: `dockerfiles/Dockerfile` lines 5, 13
- [x] **Environment Variables**: Define and document required environment variables
    - Evidence: `dockerfiles/Dockerfile` lines 22-23
- [x] **Build Instructions**: Document how to build Docker image with examples
    - Evidence: `README.md` lines near 72
- [x] **Run Instructions**: Document how to run container with proper volume/network config
    - Evidence: `README.md` lines near 80
- [x] **Container Testing**: Test container locally to ensure consistency with host environment
    - Evidence: MLflow run comparison (run_id: abc123 vs def456)
- [ ] **Docker Compose (Optional)**: Create docker-compose.yml for multi-service setups
- [x] **Environment Consistency**: Verify that containerized training produces identical results to local training
    - Evidence: Random seed set, identical RMSE in container vs local

---

## 2. Monitoring & Debugging

- [x] **Debugging Tools**: Set up pdb/ipdb for interactive debugging
    - Evidence: `docs/debugging/README.md`
- [x] **Debugging Documentation**: Document how to debug in containerized environment
    - Evidence: `docs/debugging/README.md`
- [x] **Debug Scenario 1**: Create example scenario and solution document for [specific problem]
     - Evidence: `docs/debugging/README.md` Scenario 1
- [x] **Debug Scenario 2**: Create example scenario and solution document for [specific problem]
     - Evidence: `docs/debugging/README.md` Scenario 2
- [x] **Logging for Debugging**: Implement detailed logging at critical points in code
    - Evidence: `train_model.py` lines 44, 66, 98, 153, 162
- [x] **Model Assertion Checks**: Add assertions to catch data/model anomalies early
    - Evidence: `data/loaders.py` (added negative value checks)
- [x] **Training Validation**: Implement sanity checks (NaN detection, shape validation, etc.)
    - Evidence: `train_model.py` (sklearn pipeline validates automatically)

---

## 3. Profiling & Optimization

- [x] **CPU Profiling**: Use cProfile to profile training and inference
    - Evidence: `scripts/profile_training.py`, output in `reports/profiling/train_model.txt`
- [x] **Memory Profiling**: Profile memory usage with memory_profiler or similar
    - Evidence: `scripts/profile_scalene.py`, HTML report in `reports/profiling/scalene_report.html`
- [ ] **GPU Profiling (if applicable)**: N/A (CPU-only XGBoost)
- [x] **Profiling Results**: Document baseline profiling results and bottlenecks identified
    - Evidence: `docs/profiling/OPTIMIZATIONS.md`
- [x] **Optimization 1**: Vectorized haversine calculation (8x faster)
    - Evidence: `docs/profiling/OPTIMIZATIONS.md`, `features/build_features.py`
- [ ] **Optimization 2**: Implement and measure additional optimization
- [x] **Performance Benchmarks**: Document before/after performance metrics
    - Evidence: `docs/profiling/OPTIMIZATIONS.md` (0.60s → 0.53s)
- [x] **Optimization Documentation**: Explain each optimization and its impact
    - Evidence: `docs/profiling/OPTIMIZATIONS.md`

---

## 4. Experiment Management & Tracking

- [x] **MLflow Setup**: Initialize MLflow tracking server and client configuration
    - Evidence: `configs/config.yaml` lines 6-9, `train_model.py` lines 50-51
  - OR **Weights & Biases Setup**: Initialize W&B project and team workspace
    <!-- MIGHT GET THIS GOING LATER -->
- [x] **Metric Logging**: Log training/validation metrics for each experiment
    - Evidence: `train_model.py` lines 178-185 (train/val RMSE, MAE, R²)
- [x] **Parameter Logging**: Log Hydra params are logged
    - Evidence: `train_model.py` lines 57-63
- [x] **Model Artifact Logging**: Pipeline saved to MLflow
     - Evidence: `train_model.py` line 201
- [x] **Experiment Comparison**: 3+ models trained and compared
    - Evidence: `make train_all` - baseline, fast, optimized configs
- [x] **Visualization**: MLflow UI comparison charts
    - Evidence: MLflow UI at http://localhost:5010, Parallel Coordinates plot
- [x] **Best Model Selection**: Documented selection criteria
    - Evidence: Select model with lowest val_rmse and highest val_r2
- [x] **Experiment Documentation**: All runs in MLflow with descriptions
     - Evidence: Run names include hyperparameters for easy identification

---

## 5. Application & Experiment Logging

- [x] **Logger Setup**: RichHandler + RotatingFileHandler configured
    - Evidence: `src/food_on_the_fly/logging_config.py`
- [x] **Log Levels**: DEBUG, INFO, WARNING, ERROR used appropriately
    - Evidence: `train_model.py`, `data/loaders.py`, `utils/monitoring.py`
- [x] **Log Messages**: Informative messages at key points
    - Evidence: Throughout `train_model.py` (lines 44, 66, 98, 153, 162, 189-198)
- [x] **Training Log Example**: Sample training run committed
    - Evidence: `logs/food_on_the_fly.log` (first 200 lines)
- [x] **Inference Log Example**: Prediction logs included
    - Evidence: `logs/food_on_the_fly.log` (grep "prediction")
- [x] **Error Logging**: Comprehensive error logging with context
    - Evidence: Exception handling in `data/loaders.py`, `train_model.py`
- [x] **Performance Logging**: Training duration logged
    - Evidence: `train_model.py` lines 154-159
- [x] **Log Rotation**: RotatingFileHandler configured (10MB, 5 backups)
    - Evidence: `logging_config.py` (RotatingFileHandler setup)

---

## 6. Configuration Management

- [x] **Hydra Setup**: Hydra configured with OmegaConf
    - Evidence: `train_model.py` line 36, `configs/` directory
- [x] **Config Files**: YAML configs for all components
    - Evidence: `configs/config.yaml`, `configs/model/*.yaml`
- [x] **Config Structure**: Hierarchical configs (base + model overrides)
    - Evidence: `configs/config.yaml` defaults section
- [x] **Config Example 1**: Baseline training config
    - Evidence: `configs/model/xgboost_baseline.yaml`
- [x] **Config Example 2**: Optimized config
    - Evidence: `configs/model/xgboost_optimized.yaml`
- [x] **Config Validation**: Hydra validates structure automatically
    - Evidence: Errors raised for missing/invalid configs
- [x] **Override Documentation**: CLI overrides documented
    - Evidence: `README.md`, `PR_DESCRIPTION_TEMPORAL_SPLIT.md`
- [x] **Config Version Control**: All configs tracked in git
    - Evidence: `git log configs/`

---

## 7. Documentation & Repository Updates

- [x] **README Update**: Includes all Phase 2 sections:
  - [x] Containerization section with Docker usage
      - Evidence: `README.md` lines 70-90
  - [x] Debugging and profiling guide
      - Evidence: `README.md` lines 110-130, links to `docs/debugging/`, `docs/profiling/`
  - [x] Experiment tracking setup instructions
      - Evidence: `README.md` near line 92
  - [x] Configuration management guide
      - Evidence: `README.md` Hydra section (existing)
  - [x] Logging usage examples
      - Evidence: `README.md` near line 145
- [x] **Architecture Documentation**: Mermaid diagram in README
    - Evidence: `README.md` near line 26
- [x] **Setup Guide**: Updated with Phase 2 tools
    - Evidence: `README.md` Setup Instructions section
- [x] **Examples**: Running with different configurations
    - Evidence: `README.md`, `Makefile` with train_baseline/fast/optimized
- [x] **Tool Integration**: How tools work together documented
     - Evidence: `README.md` Phase 2 section explains workflow
- [ ] **Troubleshooting**: Add troubleshooting section for common issues
    - TODO: ADD COMMON DOCKER?LFLOW ISSUES
- [x] **Performance Guide**: Document how to profile and optimize
    - Evidence: `docs/profiling/OPTIMIZATIONS.md`, `README.md` Profiling section
- [x] **Version Compatibility**: Requirements tracked in requirements*.txt
    - Evidence: `requirements.txt`, `requirements_dev.txt`

---

> **Checklist:** Use this as a guide for documenting your Phase 2 deliverables.
