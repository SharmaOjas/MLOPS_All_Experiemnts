"""Checks against the real trained model and the real FD001 test set."""
import json

import numpy as np
import pytest

from src.config import path
from src.data import load_split, load_test_rul
from src.metrics import regression_report
from src.predict import InputError

pytestmark = pytest.mark.model


def test_prediction_in_valid_range(predictor, test_engine_history):
    out = predictor.predict_history(test_engine_history)
    assert 0 <= out["predicted_rul"] <= predictor.rul_cap
    assert out["last_cycle"] == test_engine_history["cycle"].max()


def test_single_and_fleet_predictions_agree(predictor, test_engine_history):
    single = predictor.predict_history(test_engine_history)["predicted_rul"]
    fleet = predictor.predict_fleet(test_engine_history)["predicted_rul"].iloc[0]
    assert np.isclose(single, fleet, atol=0.01)


def test_model_beats_baseline_on_official_test_set(predictor):
    """Recompute test RMSE from scratch; must match saved metadata and beat predicting the mean."""
    pred = predictor.predict_fleet(load_split("test")).set_index("unit")["predicted_rul"]
    truth = load_test_rul().loc[pred.index]
    rep = regression_report(truth, pred)
    saved = json.loads((path("models") / "metadata.json").read_text())["metrics"]
    assert np.isclose(rep["rmse"], saved["test_rmse"], atol=0.05)
    assert rep["rmse"] < truth.std()
    assert rep["r2"] > 0.6


def test_rejects_missing_sensor(predictor, test_engine_history):
    with pytest.raises(InputError, match="Missing"):
        predictor.predict_history(test_engine_history.drop(columns=[predictor.sensors[0]]))


def test_rejects_non_numeric(predictor, test_engine_history):
    bad = test_engine_history.astype({"s2": object}).copy()
    bad.iloc[0, bad.columns.get_loc("s2")] = "abc"
    with pytest.raises(InputError):
        predictor.predict_history(bad)
