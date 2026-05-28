# Debugging Guide - Food on the Fly

## Installation

```bash
# Install ipdb (enhanced pdb with syntax highlighting)
pip install ipdb
```

## Basic pdb/ipdb Commands

| Command | Shortcut | Description |
|---------|----------|-------------|
| `next` | `n` | Execute current line, step over functions |
| `step` | `s` | Execute current line, step into functions |
| `continue` | `c` | Continue execution until next breakpoint |
| `print <var>` | `p <var>` | Print value of variable |
| `list` | `l` | Show current code context |
| `where` | `w` | Show stack trace |
| `up` | `u` | Move up one stack frame |
| `down` | `d` | Move down one stack frame |
| `break <line>` | `b <line>` | Set breakpoint at line number |
| `quit` | `q` | Exit debugger |

## How to Add Breakpoints

**Method 1: Inline breakpoint**
```python
import pdb; pdb.set_trace()  # Execution pauses here
```

**Method 2: Post-mortem debugging**
```python
import pdb
try:
    risky_operation()
except Exception:
    pdb.post_mortem()  # Debug at point of exception
```

**Method 3: Command-line debugging**
```bash
python -m pdb -m food_on_the_fly.train_model
```

---

## Debugging Scenario 1: Model Training Fails with NaN Loss

### Problem Description
Training fails mid-run with error:
```
RuntimeError: Loss is NaN. Check your data for inf/nan values.
```

### Debug Steps

**1. Add breakpoint before model training:**

In `src/food_on_the_fly/train_model.py`, line 156:
```python
logger.info("Training XGBoost model...")
train_start = time.perf_counter()

# ADD THIS LINE:
import ipdb; ipdb.set_trace()

with SystemMetricsLogger(interval_seconds=0.1):
    pipeline.fit(X_train, y_train)
```

**2. Run training:**
```bash
make train
```

**3. When debugger starts, inspect data:**
```python
# Check for NaN in features
(ipdb) X_train.isna().sum()

# Check for NaN in target
(ipdb) y_train.isna().sum()

# Check for infinite values
(ipdb) import numpy as np
(ipdb) np.isinf(X_train.select_dtypes(include=[np.number])).sum()

# Check target statistics
(ipdb) y_train.describe()
```

**4. Identify the issue:**
```python
(ipdb) y_train[y_train < 0]
# Output: Found 3 rows with negative delivery times!
```

### Solution
Add validation in data loading:

```python
# In src/food_on_the_fly/data/loaders.py
def load_processed(filename: str) -> pd.DataFrame:
    df = pd.read_csv(path)

    # ADD VALIDATION:
    if 'Time_taken (min)' in df.columns:
        invalid = df['Time_taken (min)'] < 0
        if invalid.any():
            logger.warning(f"Removing {invalid.sum()} rows with negative delivery times")
            df = df[~invalid]

    return df
```

---

## Debugging Scenario 2: All Predictions Are Identical

### Problem Description
Model trains successfully but predictions are all the same value:
```python
y_pred = [10.5, 10.5, 10.5, ..., 10.5]  # All identical!
```

### Debug Steps

**1. Add breakpoint after prediction:**

In `src/food_on_the_fly/train_model.py`, line 164:
```python
y_val_pred = pipeline.predict(X_val)

# ADD THIS LINE:
import ipdb; ipdb.set_trace()

# Calculate metrics
```

**2. Inspect predictions:**
```python
# Check prediction distribution
(ipdb) np.unique(y_val_pred)
# Output: array([10.5])  # Only one unique value!

# Check feature variance
(ipdb) X_val.std()

# Check if preprocessing broke features
(ipdb) X_val_transformed = pipeline.named_steps['preprocessor'].transform(X_val)
(ipdb) X_val_transformed.std(axis=0)
```

**3. Identify the issue:**
```python
(ipdb) X_val_transformed.std(axis=0)
# Output: All zeros! Features have no variance!
```

### Solution
Feature engineering bug - check that transformers are working:

```python
# Test haversine transformer separately
(ipdb) haversine_transformer.transform(X_val[location_features])

# Found bug: Haversine returning zeros for all rows
# Fix: Update haversine calculation formula
```

---

## Debugging in Docker

### Method 1: Interactive Container

```bash
# Run container in interactive mode
docker run -it --entrypoint /bin/bash food-on-the-fly:latest

# Inside container:
pip install ipdb
python -m ipdb -m food_on_the_fly.train_model
```

### Method 2: Attach to Running Container

```bash
# Terminal 1: Run training
docker run --name debug-train food-on-the-fly:latest

# Terminal 2: Attach to container
docker exec -it debug-train /bin/bash
pip install ipdb
# Find process ID and attach debugger
```

### Method 3: VSCode Remote Debugging

1. Add to `launch.json`:
```json
{
    "name": "Docker: Train Model",
    "type": "python",
    "request": "attach",
    "connect": {
        "host": "localhost",
        "port": 5678
    },
    "pathMappings": [
        {
            "localRoot": "${workspaceFolder}",
            "remoteRoot": "/app"
        }
    ]
}
```

2. Update Dockerfile to install debugpy:
```dockerfile
RUN pip install debugpy
CMD ["python", "-m", "debugpy", "--listen", "0.0.0.0:5678", "--wait-for-client", "-m", "food_on_the_fly.train_model"]
```

3. Run container with port forwarding:
```bash
docker run -p 5678:5678 food-on-the-fly:latest
```

---

## Debugging Best Practices

1. **Use logging first** - Often faster than interactive debugging
2. **Add assertions** - Catch bugs early with data validation
3. **Isolate the problem** - Test components individually
4. **Reproduce in notebook** - Jupyter makes iteration faster
5. **Check data shapes** - Most ML bugs are shape mismatches

## Common XGBoost Issues

| Symptom | Common Cause | Fix |
|---------|--------------|-----|
| NaN predictions | NaN in training data | Add validation |
| All same prediction | Zero variance features | Check preprocessing |
| Very slow training | Too many features | Feature selection |
| Poor generalization | Data leakage | Check temporal split |
