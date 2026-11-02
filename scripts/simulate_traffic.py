"""Send real FD001 test-engine histories to the API to generate monitoring data.

For each request a random test engine is truncated at a random cycle; its true RUL at that
cycle is known from RUL_FD001.txt, so the script also reports ground truth to /feedback.

  python scripts/simulate_traffic.py --n 100
  python scripts/simulate_traffic.py --n 50 --error-rate 0.2 --drift 3.0   # inject bad requests + sensor shift
"""
import argparse
import random
import sys
import time
from pathlib import Path

import requests

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.data import load_split, load_test_rul  # noqa: E402


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--url", default="http://127.0.0.1:8000")
    p.add_argument("--n", type=int, default=100)
    p.add_argument("--error-rate", type=float, default=0.05, help="fraction of deliberately invalid requests")
    p.add_argument("--drift", type=float, default=0.0, help="shift sensors by this many std devs")
    p.add_argument("--sleep", type=float, default=0.1)
    a = p.parse_args()

    test, truth = load_split("test"), load_test_rul()
    std = test.filter(regex=r"^s\d+$").std()
    rng = random.Random(42)
    ok = err = 0
    for i in range(a.n):
        unit = rng.choice(list(truth.index))
        hist = test[test.unit == unit]
        cut = rng.randint(min(15, len(hist)), len(hist))
        h = hist.iloc[:cut].drop(columns=["unit"]).copy()
        actual = float(truth[unit] + hist["cycle"].max() - h["cycle"].max())
        if a.drift:
            h[std.index] += a.drift * std
        if rng.random() < a.error_rate:
            h = h.drop(columns=["s2"])  # invalid: missing sensor -> HTTP 400
        try:
            r = requests.post(f"{a.url}/predict", json={"engine_id": f"FD001-{unit}", "records": h.to_dict("records")},
                              timeout=10)
        except requests.RequestException as e:
            print("Request failed:", e)
            err += 1
            continue
        if r.ok:
            ok += 1
            pred = r.json()["predicted_rul"]
            requests.post(f"{a.url}/feedback", json={"engine_id": str(unit), "predicted_rul": pred,
                                                    "actual_rul": actual}, timeout=10)
            print(f"[{i + 1}/{a.n}] engine {unit:3d} cycle {int(h.cycle.max()):3d}  pred={pred:6.1f}  true={actual:6.1f}")
        else:
            err += 1
            print(f"[{i + 1}/{a.n}] HTTP {r.status_code}: {r.text[:80]}")
        time.sleep(a.sleep)
    print(f"Done: {ok} ok, {err} errors")


if __name__ == "__main__":
    main()
