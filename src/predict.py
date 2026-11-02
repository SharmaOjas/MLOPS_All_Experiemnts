"""Inference: sensor history of one engine -> predicted RUL at its last cycle."""
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from src.config import load_config, path
from src.features import build_features, feature_columns, required_input_columns


class InputError(ValueError):
    pass


def default_model_path() -> Path:
    c = load_config()["artifacts"]
    return path(c["model_dir"]) / c["model_file"]


class RULPredictor:
    def __init__(self, model_path: str | Path | None = None):
        p = Path(model_path or default_model_path())
        if not p.exists():
            raise FileNotFoundError(f"No trained model at {p}. Run: python -m src.train")
        bundle = joblib.load(p)
        self.pipeline = bundle["pipeline"]
        self.sensors: list[str] = bundle["sensors"]
        self.window: int = bundle["window"]
        self.rul_cap: int = bundle["rul_cap"]
        self.meta: dict = bundle["meta"]
        self.drift_reference: dict = bundle["drift_reference"]
        self.features = feature_columns(self.sensors)

    def prepare(self, history: pd.DataFrame) -> pd.DataFrame:
        """Validate one engine's history; returns it sorted by cycle with a unit column."""
        need = required_input_columns(self.sensors)
        missing = [c for c in need if c not in history.columns]
        if missing:
            raise InputError(f"Missing required columns: {missing}")
        h = history[need].apply(pd.to_numeric, errors="coerce")
        if h.empty:
            raise InputError("Sensor history is empty")
        if h.isna().any().any() or not np.isfinite(h.to_numpy(float)).all():
            raise InputError("All values must be finite numbers")
        if h["cycle"].duplicated().any():
            raise InputError("Duplicate cycle numbers")
        h = h.sort_values("cycle").reset_index(drop=True)
        h.insert(0, "unit", 0)
        return h

    def predict_history(self, history: pd.DataFrame) -> dict:
        h = self.prepare(history)
        feats = build_features(h, self.sensors, self.window).tail(1)
        rul = float(np.clip(self.pipeline.predict(feats[self.features])[0], 0, self.rul_cap))
        return {
            "predicted_rul": round(rul, 2),
            "last_cycle": int(h["cycle"].iloc[-1]),
            "cycles_observed": int(len(h)),
            "model_name": self.meta["model_name"],
            "model_version": self.meta["model_version"],
        }

    def predict_fleet(self, df: pd.DataFrame) -> pd.DataFrame:
        """Batch prediction for many engines (column `unit`), one row per engine."""
        feats = build_features(df, self.sensors, self.window)
        last = feats.groupby("unit").tail(1)
        pred = np.clip(self.pipeline.predict(last[self.features]), 0, self.rul_cap)
        return pd.DataFrame({"unit": last["unit"].to_numpy(), "last_cycle": last["cycle"].to_numpy(), "predicted_rul": pred})
