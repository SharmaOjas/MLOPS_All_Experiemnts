"""Extract + validate the NASA C-MAPSS data and build RUL labels.

Each raw file is space separated with 26 columns:
unit, cycle, 3 operational settings, 21 sensors.
"""
import io
import logging
import urllib.request
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

from src.config import load_config, path

log = logging.getLogger(__name__)

OP_COLS = ["op1", "op2", "op3"]
SENSOR_COLS = [f"s{i}" for i in range(1, 22)]
COLUMNS = ["unit", "cycle", *OP_COLS, *SENSOR_COLS]

# Human readable names from Saxena et al. (2008), "Damage Propagation Modeling..."
SENSOR_NAMES = {
    "s1": "Fan inlet temperature (T2)", "s2": "LPC outlet temperature (T24)",
    "s3": "HPC outlet temperature (T30)", "s4": "LPT outlet temperature (T50)",
    "s5": "Fan inlet pressure (P2)", "s6": "Bypass-duct pressure (P15)",
    "s7": "HPC outlet pressure (P30)", "s8": "Physical fan speed (Nf)",
    "s9": "Physical core speed (Nc)", "s10": "Engine pressure ratio (epr)",
    "s11": "HPC outlet static pressure (Ps30)", "s12": "Fuel flow / Ps30 (phi)",
    "s13": "Corrected fan speed (NRf)", "s14": "Corrected core speed (NRc)",
    "s15": "Bypass ratio (BPR)", "s16": "Burner fuel-air ratio (farB)",
    "s17": "Bleed enthalpy (htBleed)", "s18": "Demanded fan speed (Nf_dmd)",
    "s19": "Demanded corrected fan speed (PCNfR_dmd)",
    "s20": "HPT coolant bleed (W31)", "s21": "LPT coolant bleed (W32)",
}


class DataValidationError(ValueError):
    pass


def raw_dir() -> Path:
    return path(load_config()["data"]["raw_dir"])


def download_dataset(force: bool = False) -> Path:
    """Download the official zip and extract the inner CMAPSSData.zip text files."""
    out = raw_dir()
    if (out / "train_FD001.txt").exists() and not force:
        log.info("Dataset already present in %s", out)
        return out
    url = load_config()["data"]["download_url"]
    log.info("Downloading %s", url)
    with urllib.request.urlopen(url, timeout=120) as r:
        outer = zipfile.ZipFile(io.BytesIO(r.read()))
    inner_name = next(n for n in outer.namelist() if n.endswith("CMAPSSData.zip"))
    inner = zipfile.ZipFile(io.BytesIO(outer.read(inner_name)))
    out.mkdir(parents=True, exist_ok=True)
    for name in inner.namelist():
        base = Path(name).name  # flatten; ignores any directory parts in the archive
        if base.endswith((".txt", ".pdf")):
            (out / base).write_bytes(inner.read(name))
    return out


def load_split(split: str, subset: str | None = None) -> pd.DataFrame:
    """Load train_/test_ file for a subset (FD001..FD004)."""
    subset = subset or load_config()["data"]["subset"]
    f = raw_dir() / f"{split}_{subset}.txt"
    if not f.exists():
        raise FileNotFoundError(f"{f} not found. Run: python -m src.data --download")
    df = pd.read_csv(f, sep=r"\s+", header=None, names=COLUMNS)
    return validate(df)


def load_test_rul(subset: str | None = None) -> pd.Series:
    """True RUL at the last observed cycle of each test engine (index = unit)."""
    subset = subset or load_config()["data"]["subset"]
    rul = pd.read_csv(raw_dir() / f"RUL_{subset}.txt", sep=r"\s+", header=None, names=["rul"])["rul"]
    rul.index = np.arange(1, len(rul) + 1)
    rul.index.name = "unit"
    return rul


def validate(df: pd.DataFrame) -> pd.DataFrame:
    """Schema and sanity checks. Returns a cleaned copy sorted by unit/cycle."""
    missing = [c for c in COLUMNS if c not in df.columns]
    if missing:
        raise DataValidationError(f"Missing columns: {missing}")
    df = df[COLUMNS].copy()
    for c in COLUMNS:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    if df.empty:
        raise DataValidationError("No rows")
    bad = df.isna().sum()
    if bad.any():
        raise DataValidationError(f"Non-numeric or missing values: {bad[bad > 0].to_dict()}")
    if not np.isfinite(df.to_numpy(dtype=float)).all():
        raise DataValidationError("Infinite values present")
    if (df["cycle"] < 1).any():
        raise DataValidationError("cycle must be >= 1")
    if df.duplicated(["unit", "cycle"]).any():
        raise DataValidationError("Duplicate (unit, cycle) rows")
    df[["unit", "cycle"]] = df[["unit", "cycle"]].astype(int)
    return df.sort_values(["unit", "cycle"]).reset_index(drop=True)


def add_train_rul(df: pd.DataFrame, cap: int | None) -> pd.DataFrame:
    """Training engines run to failure: RUL = last cycle - current cycle."""
    df = df.copy()
    df["rul"] = df.groupby("unit")["cycle"].transform("max") - df["cycle"]
    if cap:
        df["rul"] = df["rul"].clip(upper=cap)
    return df


def add_test_rul(df: pd.DataFrame, rul_at_end: pd.Series) -> pd.DataFrame:
    """Test engines stop before failure: RUL = given RUL + (last observed cycle - cycle)."""
    df = df.copy()
    last = df.groupby("unit")["cycle"].transform("max")
    df["rul"] = df["unit"].map(rul_at_end) + last - df["cycle"]
    if df["rul"].isna().any():
        raise DataValidationError("Test units without RUL label")
    return df


def data_summary(df: pd.DataFrame) -> dict:
    life = df.groupby("unit")["cycle"].max()
    return {
        "rows": int(len(df)), "engines": int(df["unit"].nunique()),
        "min_cycles": int(life.min()), "max_cycles": int(life.max()),
        "mean_cycles": float(life.mean()),
    }


if __name__ == "__main__":
    import argparse

    logging.basicConfig(level=logging.INFO)
    p = argparse.ArgumentParser()
    p.add_argument("--download", action="store_true")
    p.add_argument("--force", action="store_true")
    a = p.parse_args()
    if a.download:
        print("Dataset at", download_dataset(force=a.force))
    for split in ("train", "test"):
        print(split, data_summary(load_split(split)))
