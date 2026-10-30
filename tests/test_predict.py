import numpy as np
import pandas as pd
import pytest

from churniq.predict import predict_customers


class FakeModel:
    def predict_proba(self, X):
        probabilities = np.full(len(X), 0.7)
        return np.column_stack([1 - probabilities, probabilities])


def test_predict_customers_returns_expected_outputs():
    X = pd.DataFrame({"feature": [1, 2]})

    result = predict_customers(
        FakeModel(),
        X,
        threshold=0.5,
        expected_features=["feature"],
    )

    assert list(result.columns) == [
        "predicted_churn_probability",
        "predicted_churn",
    ]
    assert result["predicted_churn"].tolist() == ["Yes", "Yes"]
    assert np.allclose(
        result["predicted_churn_probability"].to_numpy(),
        [0.7, 0.7],
    )


def test_rejects_invalid_threshold():
    X = pd.DataFrame({"feature": [1]})

    with pytest.raises(ValueError, match="threshold"):
        predict_customers(FakeModel(), X, threshold=1.5)


def test_rejects_empty_input():
    X = pd.DataFrame({"feature": []})

    with pytest.raises(ValueError, match="at least one row"):
        predict_customers(FakeModel(), X)


def test_rejects_missing_features():
    X = pd.DataFrame({"wrong_feature": [1]})

    with pytest.raises(ValueError, match="Missing required features"):
        predict_customers(
            FakeModel(),
            X,
            expected_features=["feature"],
        )


def test_uses_model_feature_order_when_not_explicitly_supplied():
    class SchemaModel:
        feature_names_in_ = np.array(["first", "second"])

        def predict_proba(self, X):
            assert list(X.columns) == ["first", "second"]
            probabilities = np.full(len(X), 0.6)
            return np.column_stack([1 - probabilities, probabilities])

    X = pd.DataFrame({"second": [2], "first": [1]})

    result = predict_customers(SchemaModel(), X, threshold=0.5)

    assert result.loc[0, "predicted_churn"] == "Yes"


def test_rejects_unexpected_feature_when_model_schema_is_available():
    class SchemaModel:
        feature_names_in_ = np.array(["feature"])

        def predict_proba(self, X):
            probabilities = np.full(len(X), 0.7)
            return np.column_stack([1 - probabilities, probabilities])

    X = pd.DataFrame({"feature": [1], "unexpected": [2]})

    with pytest.raises(ValueError, match="Unexpected features"):
        predict_customers(SchemaModel(), X)


def test_rejects_non_finite_threshold():
    X = pd.DataFrame({"feature": [1]})

    with pytest.raises(ValueError, match="threshold"):
        predict_customers(FakeModel(), X, threshold=float("nan"))


def test_rejects_duplicate_input_columns():
    X = pd.DataFrame([[1, 2]], columns=["feature", "feature"])

    with pytest.raises(ValueError, match="Duplicate input column"):
        predict_customers(FakeModel(), X)
