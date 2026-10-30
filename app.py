from pathlib import Path
import sys

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from churniq.predict import load_prediction_bundle, predict_customers

MODEL_PATH = ROOT / "artifacts" / "churniq_logistic_regression.joblib"
METADATA_PATH = ROOT / "artifacts" / "model_metadata.json"

st.set_page_config(
    page_title="CHURNIQ | Customer Churn Prediction",
    page_icon="📊",
    layout="wide",
)

st.title("CHURNIQ")
st.subheader("Customer Churn Prediction")
st.write(
    "Estimate churn probability from a customer profile. "
    "This is a decision-support estimate, not a guarantee."
)

@st.cache_resource
def get_model_bundle():
    return load_prediction_bundle(MODEL_PATH, METADATA_PATH)

if not MODEL_PATH.is_file() or not METADATA_PATH.is_file():
    st.error("Model artifacts are missing. Restore the saved model before running the app.")
    st.stop()

try:
    model, metadata, threshold, expected_features = get_model_bundle()
except Exception as exc:
    st.error(f"Could not load the saved model: {exc}")
    st.stop()

st.caption(
    f"Model: {metadata.get('model_type', 'Saved Logistic Regression pipeline')} "
    f"| Decision threshold: {threshold:.3f}"
)

with st.form("customer_profile"):
    st.markdown("### Customer profile")
    left, right = st.columns(2)

    with left:
        gender = st.selectbox("Gender", ["Female", "Male"])
        senior = st.selectbox("Senior citizen", [0, 1], format_func=lambda x: "Yes" if x else "No")
        partner = st.selectbox("Has partner?", ["Yes", "No"])
        dependents = st.selectbox("Has dependents?", ["Yes", "No"])
        tenure = st.slider("Tenure (months)", 0, 72, 12)
        phone = st.selectbox("Phone service", ["Yes", "No"])
        multiple_lines = st.selectbox(
            "Multiple lines",
            ["No", "Yes", "No phone service"],
        )
        internet = st.selectbox(
            "Internet service",
            ["DSL", "Fiber optic", "No"],
        )
        online_security = st.selectbox(
            "Online security",
            ["No", "Yes", "No internet service"],
        )
        online_backup = st.selectbox(
            "Online backup",
            ["No", "Yes", "No internet service"],
        )

    with right:
        device_protection = st.selectbox(
            "Device protection",
            ["No", "Yes", "No internet service"],
        )
        tech_support = st.selectbox(
            "Tech support",
            ["No", "Yes", "No internet service"],
        )
        streaming_tv = st.selectbox(
            "Streaming TV",
            ["No", "Yes", "No internet service"],
        )
        streaming_movies = st.selectbox(
            "Streaming movies",
            ["No", "Yes", "No internet service"],
        )
        contract = st.selectbox(
            "Contract",
            ["Month-to-month", "One year", "Two year"],
        )
        paperless = st.selectbox("Paperless billing", ["Yes", "No"])
        payment = st.selectbox(
            "Payment method",
            [
                "Electronic check",
                "Mailed check",
                "Bank transfer (automatic)",
                "Credit card (automatic)",
            ],
        )
        monthly_charges = st.number_input(
            "Monthly charges",
            min_value=0.0,
            max_value=200.0,
            value=65.0,
            step=1.0,
        )
        total_charges = st.number_input(
            "Total charges",
            min_value=0.0,
            max_value=10000.0,
            value=780.0,
            step=10.0,
        )

    submitted = st.form_submit_button(
        "Estimate churn",
        type="primary",
        use_container_width=True,
    )

if submitted:
    customer = {
        "gender": gender,
        "SeniorCitizen": senior,
        "Partner": partner,
        "Dependents": dependents,
        "tenure": tenure,
        "PhoneService": phone,
        "MultipleLines": multiple_lines,
        "InternetService": internet,
        "OnlineSecurity": online_security,
        "OnlineBackup": online_backup,
        "DeviceProtection": device_protection,
        "TechSupport": tech_support,
        "StreamingTV": streaming_tv,
        "StreamingMovies": streaming_movies,
        "Contract": contract,
        "PaperlessBilling": paperless,
        "PaymentMethod": payment,
        "MonthlyCharges": monthly_charges,
        "TotalCharges": total_charges,
    }

    try:
        customer_df = pd.DataFrame(
            [{feature: customer[feature] for feature in expected_features}]
        )
        prediction = predict_customers(
            model,
            customer_df,
            threshold=threshold,
            expected_features=expected_features,
        ).iloc[0]

        probability = float(prediction["predicted_churn_probability"])
        label = str(prediction["predicted_churn"])

        st.markdown("### Prediction")
        metric_col1, metric_col2 = st.columns(2)
        metric_col1.metric("Estimated churn probability", f"{probability:.1%}")
        metric_col2.metric("Threshold-based prediction", label)

        st.progress(probability)

        if label == "Yes":
            st.warning(
                "The estimated probability meets or exceeds the model's "
                "recorded decision threshold."
            )
        else:
            st.success(
                "The estimated probability is below the model's "
                "recorded decision threshold."
            )

        st.caption(
            "Predictions depend on the input profile and historical data. "
            "Do not use this result as the sole basis for customer decisions."
        )

    except Exception as exc:
        st.error(f"Prediction failed: {exc}")

with st.expander("About this model"):
    st.write(
        "CHURNIQ uses a saved scikit-learn Logistic Regression pipeline. "
        "Its recorded threshold was selected using validation data. "
        "Performance on new customer populations may differ."
    )
