# Dataset

**Dataset:** IBM Telco Customer Churn  
**Expected file:** `WA_Fn-UseC_-Telco-Customer-Churn.csv`

## Setup

Place the CSV file in this directory for local training.

## Contents and Preprocessing

The dataset contains customer service, billing, tenure, and churn fields. `customerID` is excluded from model features.

The training pipeline converts `TotalCharges` to numeric and assigns zero to blank charges only when the affected customers have zero tenure. Unexpected missing charges trigger an error for review.

## Source and Licensing

The dataset is identified as IBM Telco Customer Churn. The exact original source URL, applicable license, redistribution permissions, and attribution requirements have not yet been verified.

Before redistributing the dataset, verify the original source's license, terms, and attribution requirements. This repository does not claim ownership of the dataset.

Do not commit the CSV until its redistribution terms have been checked. Obtain the dataset from an authorized source and record the verified source URL and applicable terms here.
