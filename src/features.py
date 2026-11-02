"""Feature engineering on engine sensor histories + a file-based feature store.

Every feature at cycle t uses only cycles <= t of the SAME engine (causal,
grouped by unit). The same `build_features` call is used for training,
offline evaluation, the API and the dashboard, so there is no train/serve skew.
"""
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from src.data import SENSOR_COLS

FEATURE_VERSION = "v1"


def select_sensors(train: pd.DataFrame, min_std: float) -> list[str]:
    """Keep sensors that actually vary in training data (FD001 has 7 flat ones)."""
    std = train[SENSOR_COLS].std()
    return [s for s in SENSOR_COLS if std[s] > min_std]


def build_features(df: pd.DataFrame, sensors: list[str], window: int) -> pd.DataFrame:
    """Return unit, cycle, (rul), and engineered features, row-aligned with df."""
    df = df.sort_values(["unit", "cycle"]).reset_index(drop=True)
    g = df.groupby("unit", sort=False)
    out = {"unit": df["unit"], "cycle": df["cycle"]}
    if "rul" in df:
        out["rul"] = df["rul"]

    x = df["cycle"].astype(float)
    pos = g.cumcount()  # 0-based position inside each engine's history

    def roll_mean(s: pd.Series) -> pd.Series:
        return s.rolling(window, min_periods=1).mean()  # BUG: not grouped by engine -> leaks across units

    mx = roll_mean(x)
    var_x = roll_mean(x * x) - mx**2

    for s in sensors:
        y = df[s].astype(float)
        my = roll_mean(y)
        out[s] = y
        out[f"{s}_mean"] = my
        out[f"{s}_std"] = (
            y.groupby(df["unit"], sort=False).rolling(window, min_periods=2).std()
            .reset_index(level=0, drop=True).fillna(0.0)
        )
        # Least-squares slope over the window: cov(x, y) / var(x)
        cov = roll_mean(x * y) - mx * my
        out[f"{s}_slope"] = np.where(var_x > 1e-9, cov / var_x.where(var_x > 1e-9, 1.0), 0.0)
        # Drift from this engine's own early-life baseline (mean of first `window` cycles,
        # computed causally so early rows never see later rows)
        base = (g[s].cumsum() / (pos + 1)).where(pos < window)
        base = base.groupby(df["unit"], sort=False).ffill()
        out[f"{s}_delta"] = y - base

    return pd.DataFrame(out)


def feature_columns(sensors: list[str]) -> list[str]:
    cols = ["cycle"]
    for s in sensors:
        cols += [s, f"{s}_mean", f"{s}_std", f"{s}_slope", f"{s}_delta"]
    return cols


def last_cycle_rows(features: pd.DataFrame) -> pd.DataFrame:
    """One row per engine: its latest observed cycle (what RUL is predicted from)."""
    return features.groupby("unit", sort=False).tail(1).reset_index(drop=True)


def required_input_columns(sensors: list[str]) -> list[str]:
    return ["cycle", *sensors]


class FeatureStore:
    """Parquet tables + JSON metadata (version, schema, row count, content hash)."""

    def __init__(self, root: str | Path):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def write(self, name: str, df: pd.DataFrame, meta: dict | None = None) -> dict:
        df.to_parquet(self.root / f"{name}.parquet", index=False)
        info = {
            "name": name,
            "feature_version": FEATURE_VERSION,
            "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "rows": int(len(df)),
            "columns": list(df.columns),
            "sha256": hashlib.sha256(pd.util.hash_pandas_object(df, index=False).values.tobytes()).hexdigest(),
            **(meta or {}),
        }
        (self.root / f"{name}.json").write_text(json.dumps(info, indent=2))
        return info

    def read(self, name: str) -> pd.DataFrame:
        f = self.root / f"{name}.parquet"
        if not f.exists():
            raise FileNotFoundError(f"Feature table '{name}' not in {self.root}. Run: python -m src.train")
        return pd.read_parquet(f)

    def metadata(self, name: str) -> dict:
        return json.loads((self.root / f"{name}.json").read_text())

    def list(self) -> list[str]:
        return sorted(p.stem for p in self.root.glob("*.parquet"))


