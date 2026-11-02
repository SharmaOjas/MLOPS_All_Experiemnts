"""Regression metrics + the PHM08 / NASA asymmetric scoring function."""
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def nasa_score(y_true, y_pred) -> float:
    """Saxena et al. 2008. d = pred - true. Late predictions (d > 0) are penalised
    harder (exp(d/10)) than early ones (exp(-d/13)) because they risk failure. Lower is better."""
    d = np.asarray(y_pred, float) - np.asarray(y_true, float)
    return float(np.sum(np.where(d < 0, np.exp(-d / 13) - 1, np.exp(d / 10) - 1)))


def regression_report(y_true, y_pred) -> dict:
    return {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "r2": float(r2_score(y_true, y_pred)),
        "nasa_score": nasa_score(y_true, y_pred),
    }
