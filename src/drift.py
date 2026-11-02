"""Input-data drift: compare live sensor values with the training distribution.

Uses the Population Stability Index (PSI) on decile bins of the training data.
Rule of thumb: PSI < 0.1 stable, 0.1-0.25 moderate shift, > 0.25 significant shift.
This detects *input* drift only; it says nothing about prediction accuracy.
"""
import numpy as np
import pandas as pd

PSI_EPS = 1e-4


def build_reference(train: pd.DataFrame, sensors: list[str], bins: int = 10) -> dict:
    ref = {}
    for s in sensors:
        v = train[s].to_numpy(float)
        edges = np.unique(np.quantile(v, np.linspace(0, 1, bins + 1)))
        counts, _ = np.histogram(v, bins=edges)
        ref[s] = {
            "edges": edges.tolist(),
            "expected": (counts / counts.sum()).tolist(),
            "mean": float(v.mean()), "std": float(v.std()),
            "min": float(v.min()), "max": float(v.max()),
        }
    return ref


def psi(reference: dict, values) -> float:
    edges = np.asarray(reference["edges"])
    v = np.clip(np.asarray(values, float), edges[0], edges[-1])  # out-of-range -> outer bins
    actual = np.histogram(v, bins=edges)[0] / max(len(v), 1)
    expected = np.asarray(reference["expected"])
    a, e = np.maximum(actual, PSI_EPS), np.maximum(expected, PSI_EPS)
    return float(np.sum((a - e) * np.log(a / e)))


def drift_report(reference: dict, df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for s, ref in reference.items():
        v = df[s].to_numpy(float)
        rows.append({
            "sensor": s,
            "psi": psi(ref, v),
            "mean_shift_z": float((v.mean() - ref["mean"]) / ref["std"]) if ref["std"] else 0.0,
            "out_of_range_pct": float(((v < ref["min"]) | (v > ref["max"])).mean() * 100),
        })
    out = pd.DataFrame(rows)
    out["status"] = pd.cut(out["psi"], [-np.inf, 0.1, 0.25, np.inf], labels=["stable", "moderate", "significant"])
    return out
