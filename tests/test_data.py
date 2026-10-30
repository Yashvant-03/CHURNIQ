from pathlib import Path

import pandas as pd
import pytest

from churniq.train import load_dataset


def test_dataset_loader_rejects_missing_target(tmp_path):
    path = tmp_path / "bad.csv"

    pd.DataFrame({
        "customerID": ["A"],
        "TotalCharges": ["10"],
    }).to_csv(path, index=False)

    with pytest.raises(ValueError, match="Missing required columns"):
        load_dataset(path)


def test_dataset_loader_converts_blank_total_charges(tmp_path):
    path = tmp_path / "data.csv"

    pd.DataFrame({
        "customerID": ["A", "B"],
        "Churn": ["No", "Yes"],
        "tenure": [0, 1],
        "TotalCharges": [" ", "50"],
    }).to_csv(path, index=False)

    df = load_dataset(path)

    assert pd.api.types.is_numeric_dtype(df["TotalCharges"])
    assert df.loc[0, "TotalCharges"] == 0.0
    assert df.loc[1, "TotalCharges"] == 50.0


def test_threshold_selection_returns_valid_threshold():
    import numpy as np

    from churniq.train import select_threshold

    y_true = np.array([0, 0, 1, 1])
    probabilities = np.array([0.1, 0.4, 0.6, 0.9])

    threshold = select_threshold(y_true, probabilities)

    assert 0.0 <= threshold <= 1.0


def test_dataset_loader_rejects_missing_total_charges_for_active_customer(
    tmp_path,
):
    path = tmp_path / "invalid_charges.csv"

    pd.DataFrame({
        "customerID": ["A", "B"],
        "Churn": ["No", "Yes"],
        "tenure": [12, 1],
        "TotalCharges": [" ", "1"],
    }).to_csv(path, index=False)

    with pytest.raises(ValueError, match="nonzero tenure"):
        load_dataset(path)
