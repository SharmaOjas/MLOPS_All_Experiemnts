import pandas as pd
import pytest

from src.data import DataValidationError, add_test_rul, add_train_rul, validate


def test_validate_sorts_and_accepts(engines):
    out = validate(engines.sample(frac=1, random_state=1))
    assert out[["unit", "cycle"]].equals(engines[["unit", "cycle"]])


def test_validate_rejects_missing_column(engines):
    with pytest.raises(DataValidationError, match="Missing"):
        validate(engines.drop(columns=["s4"]))


def test_validate_rejects_nan_and_duplicates(engines):
    bad = engines.copy()
    bad.loc[3, "s2"] = None
    with pytest.raises(DataValidationError):
        validate(bad)
    with pytest.raises(DataValidationError, match="Duplicate"):
        validate(pd.concat([engines, engines.head(1)]))


def test_train_rul_counts_down_to_zero_and_caps(engines):
    df = add_train_rul(engines, cap=25)
    u1 = df[df.unit == 1]
    assert u1["rul"].iloc[-1] == 0
    assert u1["rul"].iloc[-2] == 1
    assert u1["rul"].max() == 25  # 30-cycle engine: raw RUL 29 capped


def test_test_rul_uses_label_file(engines):
    labels = pd.Series({1: 10, 2: 50})
    df = add_test_rul(engines, labels)
    assert df[df.unit == 1]["rul"].iloc[-1] == 10
    assert df[df.unit == 2]["rul"].iloc[-1] == 50
    assert df[df.unit == 2]["rul"].iloc[0] == 50 + 19  # 20 cycles observed


def test_test_rul_missing_label(engines):
    with pytest.raises(DataValidationError):
        add_test_rul(engines, pd.Series({1: 10}))
