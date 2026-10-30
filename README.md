# CHURNIQ — Customer Churn Prediction

## Overview

CHURNIQ estimates customer churn using Logistic Regression and
the IBM Telco Customer Churn dataset.

## Test-set evaluation

| Metric | Result |
|---|---:|
| Accuracy | 0.7771 |
| Balanced accuracy | 0.7459 |
| Precision | 0.5670 |
| Recall | 0.6791 |
| F1 | 0.6180 |
| ROC-AUC | 0.8421 |
| Average Precision | 0.6343 |
| Brier score | 0.1380 |

Selected decision threshold: 0.3883.

These are the results reported by the Colab notebook.
Independent reproduction has not yet been established.

## Confusion matrix

| Actual / Predicted | No Churn | Churn |
|---|---:|---:|
| No Churn | 841 | 194 |
| Churn | 120 | 254 |

## Architecture

CSV dataset -> validation -> preprocessing -> Logistic Regression
-> churn probability -> threshold-based prediction.

Numeric features use imputation and standardization.
Categorical features use imputation and one-hot encoding.
Preprocessing is included in the saved model pipeline.

## Repository structure

- `notebooks/`: analysis notebook
- `data/`: dataset instructions
- `src/churniq/`: training and prediction modules
- `tests/`: automated tests
- `artifacts/`: model and metadata

## Setup

Python 3.10 or newer recommended.

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python -m pip install -e .
python -m pytest -q
```

## Train

```powershell
python -m churniq.train --data data/WA_Fn-UseC_-Telco-Customer-Churn.csv --output-dir artifacts
```

## Limitations

- Results are dataset-specific and may not generalize.
- The threshold optimizes validation F1, not measured financial benefit.
- Model coefficients indicate associations, not causal effects.
- Calibration is not guaranteed.
- Verify dataset licensing before redistribution.
- No live deployment is claimed.
- Never commit credentials, secrets, or private customer data.

## Development status

The notebook produced a model, evaluation results, and metadata.
Automated package tests passed in the current Colab runtime.
Local Windows execution and independent retraining remain to be verified.
