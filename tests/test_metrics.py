import numpy as np

from src.metrics import nasa_score, regression_report


def test_nasa_score_asymmetric():
    assert nasa_score([50], [50]) == 0
    late, early = nasa_score([50], [60]), nasa_score([50], [40])
    assert np.isclose(late, np.exp(1) - 1)
    assert np.isclose(early, np.exp(10 / 13) - 1)
    assert late > early  # predicting failure too late is worse


def test_regression_report_perfect():
    r = regression_report([1, 2, 3], [1, 2, 3])
    assert r["mae"] == 0 and r["rmse"] == 0 and r["r2"] == 1
