"""Reproducible training and evaluation for CHURNIQ."""

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import (
    GridSearchCV,
    StratifiedKFold,
    train_test_split,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


SEED = 42
TARGET = "Churn"
ID_COLUMN = "customerID"


def load_dataset(csv_path):
    """Load and validate the IBM Telco churn dataset."""
    df = pd.read_csv(csv_path)

    required = {TARGET, ID_COLUMN, "tenure", "TotalCharges"}
    missing = required - set(df.columns)

    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    if df[TARGET].isna().any():
        raise ValueError("Target contains missing values.")

    for col in df.select_dtypes(include=["object", "string"]).columns:
        df[col] = df[col].str.strip()

    if set(df[TARGET].unique()) != {"Yes", "No"}:
        raise ValueError("Expected target values 'Yes' and 'No'.")

    df["TotalCharges"] = pd.to_numeric(
        df["TotalCharges"].replace("", np.nan),
        errors="coerce",
    )

    missing_charges = df["TotalCharges"].isna()
    zero_tenure = df["tenure"].eq(0)

    if missing_charges.any() and (
        missing_charges & ~zero_tenure
    ).any():
        raise ValueError(
            "Missing TotalCharges found for customers with nonzero tenure. "
            "Review the data before training."
        )

    df.loc[missing_charges & zero_tenure, "TotalCharges"] = 0.0

    if df.duplicated().any():
        raise ValueError("Duplicate rows found; review before training.")

    return df


def build_pipeline(X):
    """Build preprocessing and Logistic Regression as one pipeline."""
    numeric_features = X.select_dtypes(include=["number"]).columns.tolist()
    categorical_features = X.select_dtypes(
        include=["object", "category", "string"]
    ).columns.tolist()

    numeric_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])

    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore")),
    ])

    preprocessor = ColumnTransformer([
        ("numeric", numeric_pipeline, numeric_features),
        ("categorical", categorical_pipeline, categorical_features),
    ])

    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("model", LogisticRegression(max_iter=2000, random_state=SEED)),
    ])

    return pipeline, numeric_features, categorical_features


def select_threshold(y_true, probabilities):
    """Select the threshold with maximum F1 on validation data."""
    thresholds = np.unique(probabilities)
    thresholds = np.concatenate(([0.0], thresholds, [1.0]))

    scores = [
        f1_score(
            y_true,
            (probabilities >= threshold).astype(int),
            zero_division=0,
        )
        for threshold in thresholds
    ]

    # np.argmax deterministically chooses the first maximum.
    return float(thresholds[int(np.argmax(scores))])


def train_and_evaluate(csv_path, output_dir):
    """Train, tune, select a validation threshold, and evaluate on test data."""
    df = load_dataset(csv_path)

    X = df.drop(columns=[TARGET, ID_COLUMN])
    y = df[TARGET].map({"No": 0, "Yes": 1})

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=SEED,
        stratify=y,
    )

    X_dev, X_val, y_dev, y_val = train_test_split(
        X_train,
        y_train,
        test_size=0.20,
        random_state=SEED,
        stratify=y_train,
    )

    pipeline, numeric_features, categorical_features = build_pipeline(X)

    grid = GridSearchCV(
        pipeline,
        {
            "model__C": [0.1, 1.0, 10.0],
            "model__class_weight": [None, "balanced"],
        },
        scoring="average_precision",
        cv=StratifiedKFold(
            n_splits=5,
            shuffle=True,
            random_state=SEED,
        ),
        n_jobs=-1,
        refit=True,
    )

    grid.fit(X_dev, y_dev)

    val_probabilities = grid.best_estimator_.predict_proba(X_val)[:, 1]
    threshold = select_threshold(y_val, val_probabilities)

    threshold_predictions = (val_probabilities >= threshold).astype(int)

    validation_summary = {
        "method": "Maximum F1 on validation subset",
        "validation_f1": float(
            f1_score(y_val, threshold_predictions, zero_division=0)
        ),
        "validation_precision": float(
            precision_score(y_val, threshold_predictions, zero_division=0)
        ),
        "validation_recall": float(
            recall_score(y_val, threshold_predictions, zero_division=0)
        ),
    }

    # Refit on all training rows. The held-out test set remains untouched
    # until final evaluation.
    final_model = grid.best_estimator_
    final_model.fit(X_train, y_train)

    test_probabilities = final_model.predict_proba(X_test)[:, 1]
    test_predictions = (test_probabilities >= threshold).astype(int)

    metrics = {
        "Accuracy": float(accuracy_score(y_test, test_predictions)),
        "Balanced Accuracy": float(
            balanced_accuracy_score(y_test, test_predictions)
        ),
        "Precision": float(
            precision_score(y_test, test_predictions, zero_division=0)
        ),
        "Recall": float(
            recall_score(y_test, test_predictions, zero_division=0)
        ),
        "F1": float(f1_score(y_test, test_predictions, zero_division=0)),
        "ROC-AUC": float(roc_auc_score(y_test, test_probabilities)),
        "Average Precision": float(
            average_precision_score(y_test, test_probabilities)
        ),
        "Brier Score": float(
            brier_score_loss(y_test, test_probabilities)
        ),
    }

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    model_path = output_dir / "churniq_logistic_regression.joblib"
    metadata_path = output_dir / "model_metadata.json"

    joblib.dump(final_model, model_path)

    metadata = {
        "project": "CHURNIQ",
        "model_type": "LogisticRegression",
        "target": TARGET,
        "positive_class": 1,
        "positive_class_label": "Yes",
        "decision_threshold": threshold,
        "selected_hyperparameters": {
            key: value.item() if isinstance(value, np.generic) else value
            for key, value in grid.best_params_.items()
        },
        "threshold_selection": validation_summary,
        "test_metrics": metrics,
        "random_seed": SEED,
        "training_rows": int(len(X_train)),
        "test_rows": int(len(X_test)),
        "numeric_features": numeric_features,
        "categorical_features": categorical_features,
        "input_features": list(X.columns),
        "excluded_columns": [ID_COLUMN, TARGET],
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "limitations": [
            "Evaluation uses the IBM Telco Customer Churn dataset.",
            "Performance on new populations is not established.",
            "The selected threshold optimizes validation F1, not measured business profit.",
            "Predicted churn probabilities are estimates, not guarantees.",
        ],
    }

    with metadata_path.open("w", encoding="utf-8") as file:
        json.dump(metadata, file, indent=2)

    return metadata


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True, help="Path to the dataset CSV")
    parser.add_argument(
        "--output-dir",
        default="artifacts",
        help="Directory for model and metadata",
    )
    args = parser.parse_args()

    results = train_and_evaluate(args.data, args.output_dir)
    print(json.dumps(results, indent=2))
