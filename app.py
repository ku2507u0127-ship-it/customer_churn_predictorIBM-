import pickle
from pathlib import Path

import pandas as pd
import streamlit as st


ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "model.pkl"


@st.cache_resource
def load_artifacts():
    with MODEL_PATH.open("rb") as handle:
        return pickle.load(handle)


artifacts = load_artifacts()
model = artifacts["model"]
metrics = artifacts["metrics"]
feature_options = artifacts["feature_options"]
numeric_ranges = artifacts["numeric_ranges"]

st.set_page_config(page_title="Customer Churn Prediction", layout="wide")
st.title("Customer Churn Prediction Using Random Forest")
st.write(
    "This app predicts whether a customer is likely to churn using age, loyalty, income, and service usage details."
)

with st.sidebar:
    st.header("Model Snapshot")
    st.metric("Accuracy", f'{metrics["accuracy"]:.2%}')
    st.metric("ROC AUC", f'{metrics["roc_auc"]:.2%}')

st.subheader("Enter Customer Details")

col1, col2 = st.columns(2)

with col1:
    age = st.slider(
        "Age",
        int(numeric_ranges["Age"]["min"]),
        int(numeric_ranges["Age"]["max"]),
        int(round(numeric_ranges["Age"]["median"])),
    )
    frequent_flyer = st.selectbox("Frequent Flyer", feature_options["FrequentFlyer"])
    annual_income_class = st.selectbox("Annual Income Class", feature_options["AnnualIncomeClass"])

with col2:
    services_opted = st.slider(
        "Services Opted",
        int(numeric_ranges["ServicesOpted"]["min"]),
        int(numeric_ranges["ServicesOpted"]["max"]),
        int(round(numeric_ranges["ServicesOpted"]["median"])),
    )
    account_synced = st.selectbox(
        "Account Synced To Social Media",
        feature_options["AccountSyncedToSocialMedia"],
    )
    booked_hotel = st.selectbox("Booked Hotel Or Not", feature_options["BookedHotelOrNot"])

input_df = pd.DataFrame(
    [
        {
            "Age": age,
            "FrequentFlyer": frequent_flyer,
            "AnnualIncomeClass": annual_income_class,
            "ServicesOpted": services_opted,
            "AccountSyncedToSocialMedia": account_synced,
            "BookedHotelOrNot": booked_hotel,
        }
    ]
)

if st.button("Predict Churn", type="primary"):
    prediction = int(model.predict(input_df)[0])
    probability = float(model.predict_proba(input_df)[0][1])

    st.subheader("Prediction Result")
    if prediction == 1:
        st.error(f"This customer is likely to churn. Churn probability: {probability:.2%}")
    else:
        st.success(f"This customer is likely to stay. Churn probability: {probability:.2%}")

    st.write("Customer input summary")
    st.dataframe(input_df, use_container_width=True)

st.subheader("About the Model")
st.write(
    "The deployed model uses a Random Forest classifier with preprocessing for missing values and categorical encoding."
)
