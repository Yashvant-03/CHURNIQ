"""Prediction utilities for CHURNIQ."""

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd


def load_model(model_path):
    """Load a trusted, previously saved scikit-learn pipeline."""
    return joblib.load(model_path)


def load_metadata(metadata_path):
    """Load model metadata from JSON."""
    with Path(metadata_path).open("r", encoding="utf-8") as file:
        metadata = json.load(file)

    if not isinstance(metadata, dict):
        raise ValueError("Model metadata must be a JSON object.")

    return metadata


def load_prediction_bundle(model_path, metadata_path):
    """
    Load a trusted model and metadata, returning the model,
    metadata, selected threshold, and canonical input feature order.
    """
    model = load_model(model_path)
    metadata = load_metadata(metadata_path)

    threshold = metadata.get("decision_threshold")

    if (
        isinstance(threshold, bool)
        or not isinstance(threshold, (int, float))
        or not np.isfinite(threshold)
        or not 0 <= threshold <= 1
    ):
        raise ValueError(
            "Metadata must contain a valid decision_threshold in [0, 1]."
        )

    if not hasattr(model, "feature_names_in_"):
        raise ValueError(
            "Loaded model does not expose feature_names_in_; "
            "cannot determine the canonical input schema."
        )

    expected_features = list(model.feature_names_in_)

    if not expected_features:
        raise ValueError("Model input feature schema is empty.")

    if len(expected_features) != len(set(expected_features)):
        raise ValueError("Model input feature schema contains duplicates.")

    return model, metadata, float(threshold), expected_features


def predict_customers(
    model,
    customer_data,
    threshold=0.5,
    expected_features=None,
):
    """
    Predict churn probabilities and labels.

    Parameters
    ----------
    model:
        Fitted scikit-learn pipeline.
    customer_data:
        DataFrame containing model input features.
    threshold:
        Probability cutoff for predicting churn, from 0 to 1.
    expected_features:
        Optional ordered list of required input feature names.
        When supplied, missing and unexpected columns are rejected.

    Returns
    -------
    pandas.DataFrame
        Churn probability and predicted label for each row.
    """
    if not isinstance(customer_data, pd.DataFrame):
        raise TypeError("customer_data must be a pandas DataFrame.")

    if customer_data.empty:
        raise ValueError("customer_data must contain at least one row.")

    if (
        isinstance(threshold, bool)
        or not isinstance(threshold, (int, float, np.number))
        or not np.isfinite(threshold)
        or not 0 <= threshold <= 1
    ):
        raise ValueError("threshold must be a finite number between 0 and 1.")

    if customer_data.columns.duplicated().any():
        raise ValueError("Duplicate input column names are not allowed.")

    if expected_features is None and hasattr(model, "feature_names_in_"):
        expected_features = list(model.feature_names_in_)

    if expected_features is not None:
        expected_features = list(expected_features)

        if len(expected_features) != len(set(expected_features)):
            raise ValueError("Expected feature schema contains duplicates.")

        missing = set(expected_features) - set(customer_data.columns)
        extra = set(customer_data.columns) - set(expected_features)

        if missing:
            raise ValueError(f"Missing required features: {sorted(missing)}")

        if extra:
            raise ValueError(f"Unexpected features: {sorted(extra)}")

        customer_data = customer_data[expected_features]

    probabilities = np.asarray(model.predict_proba(customer_data))

    if (
        probabilities.ndim != 2
        or probabilities.shape[0] != len(customer_data)
        or probabilities.shape[1] != 2
    ):
        raise ValueError(
            "Model must return two class probabilities per input row."
        )

    # CHURNIQ training maps No -> 0 and Yes -> 1.
    probabilities = probabilities[:, 1]

    if not np.isfinite(probabilities).all():
        raise ValueError("Model returned non-finite probabilities.")

    if ((probabilities < 0) | (probabilities > 1)).any():
        raise ValueError("Model returned probabilities outside [0, 1].")

    predictions = np.where(probabilities >= threshold, "Yes", "No")

    return pd.DataFrame(
        {
            "predicted_churn_probability": probabilities,
            "predicted_churn": predictions,
        },
        index=customer_data.index,
    )
