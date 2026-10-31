# CHURNIQ — Customer Churn Prediction

## 1. Overview

CHURNIQ is a customer churn prediction project built with Python and scikit-learn. It uses Logistic Regression to estimate churn probability from customer account and service information.

The project includes a preprocessing pipeline, model evaluation, threshold-based predictions, automated tests, and a Streamlit demo.

**Live demo:** https://churniq-ml.streamlit.app/  
**GitHub repository:** https://github.com/Yashvant-03/CHURNIQ

The prediction is a decision-support estimate, not a guarantee of future customer behavior.

## 2. Model Evaluation

The following test-set metrics are reported by the Colab notebook and saved model metadata.

| Metric | Result |
|---|---:|
| Accuracy | 0.7771 |
| Balanced accuracy | 0.7459 |
| Precision | 0.5670 |
| Recall | 0.6791 |
| F1-score | 0.6180 |
| ROC-AUC | 0.8421 |
| Average Precision | 0.6343 |
| Brier score | 0.1380 |

**Selected decision threshold:** 0.3883

The threshold was selected to optimize validation F1. It was not optimized against measured business costs or financial outcomes.

The repository's independent retraining was reported as reproducing all eight metrics in Colab. Retain this statement only while the corresponding execution output remains available as evidence.

### Confusion Matrix

| Actual / Predicted | No Churn | Churn |
|---|---:|---:|
| No Churn | 841 | 194 |
| Churn | 120 | 254 |

The confusion matrix uses the selected decision threshold.

## 3. Architecture

```text
Customer dataset
      |
      v
Data validation
      |
      v
Train/test split
      |
      v
Preprocessing pipeline
  - Numeric imputation
  - Numeric standardization
  - Categorical imputation
  - One-hot encoding
      |
      v
Logistic Regression
      |
      v
Churn probability
      |
      v
Decision threshold
      |
      v
Predicted churn label
```

Preprocessing is included in the saved model pipeline to keep inference transformations consistent with training.

## 4. Repository Structure

```text
CHURNIQ/
├── artifacts/
│   ├── churniq_logistic_regression.joblib
│   └── model_metadata.json
├── data/
│   └── README.md
├── notebooks/
├── src/
│   └── churniq/
│       ├── train.py
│       └── predict.py
├── tests/
├── app.py
├── pyproject.toml
├── requirements.txt
└── README.md
```

## 5. Requirements

- Python 3.10 or newer, subject to dependency compatibility.
- Git.
- A virtual environment is recommended.

## 6. Local Setup — Windows PowerShell

Run commands separately from the repository root.

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip install -e .
python -m pytest -q
```

The package installation step makes the `churniq` module available to Python.

Local Windows execution has not yet been verified. The commands above are the documented setup procedure, not a claim of successful local execution.

## 7. Training

Place the dataset at the expected path after obtaining it from an authorized source.

```powershell
python -m churniq.train --data "data/WA_Fn-UseC_-Telco-Customer-Churn.csv" --output-dir artifacts
```

Run the command from the repository root with the package installed.

Training output and metrics should be recorded from the actual execution. Do not assume retraining will reproduce identical metrics unless the dataset, split, random seeds, dependencies, and training configuration are consistent.

## 8. Automated Tests

The latest recorded Colab test run passed:

```text
12 passed
```

This result applies to that recorded runtime. It does not establish that every environment or deployment configuration has been tested.

## 9. Dataset Provenance

The project uses the IBM Telco Customer Churn dataset.

Before redistribution, document:
- The original source and download location.
- The applicable license or terms of use.
- Any attribution requirements.
- Whether redistribution of the dataset is permitted.

Do not commit the dataset unless its redistribution terms permit it. See `data/README.md` for dataset acquisition instructions.

## 10. Limitations

- Evaluation results are specific to the dataset and test split.
- Performance may differ on real-world customer populations.
- The selected threshold optimizes validation F1, not measured financial benefit.
- Model coefficients represent learned associations, not causal effects.
- The Brier score provides a measure of probabilistic prediction error; it does not, by itself, establish adequate calibration.
- Predictions depend on the quality and representativeness of input data.
- The model should not be the sole basis for customer-related decisions.

## 11. Security

- Never commit credentials, access tokens, `.env` files, or private customer data.
- Keep raw datasets out of Git unless redistribution is explicitly permitted.
- Review `.gitignore` before committing new files.

## 12. Development Status

The project includes a training pipeline, prediction module, saved model artifacts, automated tests, and a Streamlit demo.

The live demo has been observed returning probability estimates and threshold-based labels for submitted customer profiles.

The latest recorded Colab test run passed 12 tests. Independent retraining was reported as verified in Colab. Local Windows execution and dataset licensing remain to be verified.
