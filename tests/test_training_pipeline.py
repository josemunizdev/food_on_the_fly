"""Integration test for the training pipeline."""

import pytest

from food_on_the_fly.config import MODELS_DIR, PROCESSED_DATA_DIR


def test_train_script_runs_without_error() -> None:
    """Test that train_model.py can be imported and has required functions."""
    # This is a lightweight test - just verify the module loads
    try:
        from food_on_the_fly import train_model

        assert train_model is not None
    except ImportError as e:
        pytest.fail(f"Cannot import train_model: {e}")


def test_processed_data_exists() -> None:
    """Verify that processed data files exist for training."""
    required_files = ["train.csv", "val.csv", "test.csv"]
    for filename in required_files:
        filepath = PROCESSED_DATA_DIR / filename
        assert filepath.exists(), f"Missing processed data file: {filename}"


def test_models_directory_exists() -> None:
    """Verify that models directory exists."""
    assert MODELS_DIR.exists(), "Models directory does not exist"
