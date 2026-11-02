import numpy as np
import pandas as pd

from src.drift import build_reference, psi
from src.features import FeatureStore, build_features, feature_columns, select_sensors

SENSORS = ["s2", "s3"]


def test_select_sensors_drops_constant(engines):
    engines["s1"] = 518.67
    assert "s1" not in select_sensors(engines, 1e-4)


def test_shape_and_no_nans(engines):
    f = build_features(engines, SENSORS, window=5)
    assert len(f) == len(engines)
    assert set(feature_columns(SENSORS)) <= set(f.columns)
    assert not f[feature_columns(SENSORS)].isna().any().any()


def test_rolling_resets_per_engine(engines):
    """First row of engine 2 must not be influenced by engine 1 (no leakage across units)."""
    f = build_features(engines, SENSORS, window=5)
    first = f[f.unit == 2].iloc[0]
    raw = engines[engines.unit == 2].iloc[0]
    assert first["s2_mean"] == raw["s2"]
    assert first["s2_std"] == 0 and first["s2_slope"] == 0 and first["s2_delta"] == 0


def test_features_are_causal(engines):
    """Features at cycle t are identical whether or not later cycles exist -> train/serve consistency."""
    full = build_features(engines, SENSORS, window=5)
    for t in (3, 12):
        part = build_features(engines[engines.cycle <= t], SENSORS, window=5)
        a = full[(full.unit == 1) & (full.cycle == t)][feature_columns(SENSORS)].to_numpy()
        b = part[(part.unit == 1) & (part.cycle == t)][feature_columns(SENSORS)].to_numpy()
        np.testing.assert_allclose(a, b)


def test_slope_matches_polyfit():
    df = pd.DataFrame({"unit": 1, "cycle": range(1, 11), "s2": [3.0 * c + 1 for c in range(1, 11)]})
    f = build_features(df, ["s2"], window=5)
    assert np.isclose(f["s2_slope"].iloc[-1], 3.0)


def test_feature_store_roundtrip(tmp_path, engines):
    store = FeatureStore(tmp_path)
    f = build_features(engines, SENSORS, window=5)
    meta = store.write("demo", f, {"sensors": SENSORS})
    assert store.list() == ["demo"]
    pd.testing.assert_frame_equal(store.read("demo"), f)
    assert store.metadata("demo")["sha256"] == meta["sha256"]


def test_psi_detects_shift():
    rng = np.random.default_rng(0)
    ref = build_reference(pd.DataFrame({"s2": rng.normal(0, 1, 5000)}), ["s2"])["s2"]
    assert psi(ref, rng.normal(0, 1, 2000)) < 0.1
    assert psi(ref, rng.normal(2, 1, 2000)) > 0.25
