import numpy as np
import pandas as pd
import pytest

from src.data import COLUMNS
from src.predict import default_model_path


def make_engines(lengths=(30, 20), seed=0) -> pd.DataFrame:
    """Small synthetic fleet in raw C-MAPSS layout (sensors drift upward with age)."""
    rng = np.random.default_rng(seed)
    rows = []
    for unit, n in enumerate(lengths, start=1):
        for c in range(1, n + 1):
            rows.append([unit, c, 0.0, 0.0, 100.0] + list(500 + c * 0.5 + rng.normal(0, 0.1, 21)))
    return pd.DataFrame(rows, columns=COLUMNS)


@pytest.fixture
def engines():
    return make_engines()


@pytest.fixture(scope="session")
def predictor():
    if not default_model_path().exists():
        pytest.skip("No trained model; run python -m src.train")
    from src.predict import RULPredictor
    return RULPredictor()


@pytest.fixture(scope="session")
def test_engine_history():
    """Real FD001 test engine #1 history (raw columns)."""
    from src.data import load_split
    try:
        df = load_split("test")
    except FileNotFoundError:
        pytest.skip("Dataset not downloaded")
    return df[df["unit"] == 1]
