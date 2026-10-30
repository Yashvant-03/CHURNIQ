# Dataset

Dataset: IBM Telco Customer Churn.

Expected file:
`WA_Fn-UseC_-Telco-Customer-Churn.csv`

Place the CSV in this directory for local training.

The dataset contains customer service, billing, tenure, and churn fields.
`customerID` is excluded from model features.

The training pipeline converts `TotalCharges` to numeric and assigns zero
to blank charges only when the affected customers have zero tenure.
Unexpected missing charges trigger an error for review.

Before redistributing the dataset, verify the original source's license,
terms, and attribution requirements. This repository does not claim ownership
of the dataset.
