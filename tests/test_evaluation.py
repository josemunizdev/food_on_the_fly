import numpy as np

from food_on_the_fly.evaluation.metrics import regression_report


def test_regression_report_perfect_predictions() -> None:
    """Perfect predictions should have zero error."""
    y_true = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    y_pred = np.array([1.0, 2.0, 3.0, 4.0, 5.0])

    report = regression_report(y_true, y_pred)

    assert report["mae"] == 0.0
    assert report["rmse"] == 0.0
    assert report["r2"] == 1.0


def test_regression_report_keys_present() -> None:
    """Report should contain all expected keys."""
    y_true = np.array([1, 2, 3, 4, 5])
    y_pred = np.array([1.1, 2.1, 2.9, 4.2, 4.8])

    report = regression_report(y_true, y_pred)

    assert "mae" in report
    assert "rmse" in report
    assert "r2" in report
    assert "mse" in report


def test_regression_report_rmse_is_sqrt_mse() -> None:
    """RMSE should equal sqrt(MSE)."""
    y_true = np.array([1, 2, 3, 4, 5])
    y_pred = np.array([1.5, 2.5, 3.5, 4.5, 5.5])

    report = regression_report(y_true, y_pred)

    assert np.isclose(report["rmse"], np.sqrt(report["mse"]))
