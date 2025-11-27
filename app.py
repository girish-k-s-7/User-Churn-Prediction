# stream_app.py

import os
import streamlit as st
import pandas as pd

from src.pipelines.prediction_pipeline import PredictionPipeline

st.set_page_config(page_title="Telco Churn Prediction", layout="centered")

st.title("Telco Customer Churn Prediction")
st.write("Fill in the customer details below to predict whether they are likely to churn.")

# Initialize prediction pipeline
pipeline = PredictionPipeline()

with st.form("churn_form"):
    st.subheader("Customer Information")

    customerID = st.text_input("Customer ID", value="C0001")

    gender = st.selectbox("Gender", ["Male", "Female"])
    senior = st.selectbox("Senior Citizen", [0, 1])
    partner = st.selectbox("Partner", ["Yes", "No"])
    dependents = st.selectbox("Dependents", ["Yes", "No"])

    tenure = st.number_input("Tenure (months)", min_value=0, max_value=1000, value=12)

    # Services
    phone_service = st.selectbox("Phone Service", ["Yes", "No"])
    multiple_lines = st.selectbox(
        "Multiple Lines",
        ["No", "Yes", "No phone service"],
    )

    internet_service = st.selectbox(
        "Internet Service",
        ["DSL", "Fiber optic", "No"],
    )

    online_security = st.selectbox(
        "Online Security",
        ["No", "Yes", "No internet service"],
    )
    online_backup = st.selectbox(
        "Online Backup",
        ["No", "Yes", "No internet service"],
    )
    device_protection = st.selectbox(
        "Device Protection",
        ["No", "Yes", "No internet service"],
    )
    tech_support = st.selectbox(
        "Tech Support",
        ["No", "Yes", "No internet service"],
    )
    streaming_tv = st.selectbox(
        "Streaming TV",
        ["No", "Yes", "No internet service"],
    )
    streaming_movies = st.selectbox(
        "Streaming Movies",
        ["No", "Yes", "No internet service"],
    )

    # Contract & billing
    contract = st.selectbox(
        "Contract",
        ["Month-to-month", "One year", "Two year"],
    )
    paperless_billing = st.selectbox(
        "Paperless Billing",
        ["Yes", "No"],
    )
    payment_method = st.selectbox(
        "Payment Method",
        [
            "Electronic check",
            "Mailed check",
            "Bank transfer (automatic)",
            "Credit card (automatic)",
        ],
    )

    monthly_charges = st.number_input(
        "Monthly Charges",
        min_value=0.0,
        max_value=10000.0,
        value=70.0,
        step=1.0,
    )
    total_charges = st.number_input(
        "Total Charges",
        min_value=0.0,
        max_value=1000000.0,
        value=1500.0,
        step=10.0,
    )

    submitted = st.form_submit_button("Predict Churn")

if submitted:
    # Build input row matching training features (all except 'Churn')
    input_dict = {
        "customerID": customerID,
        "gender": gender,
        "SeniorCitizen": senior,
        "Partner": partner,
        "Dependents": dependents,
        "tenure": tenure,
        "PhoneService": phone_service,
        "MultipleLines": multiple_lines,
        "InternetService": internet_service,
        "OnlineSecurity": online_security,
        "OnlineBackup": online_backup,
        "DeviceProtection": device_protection,
        "TechSupport": tech_support,
        "StreamingTV": streaming_tv,
        "StreamingMovies": streaming_movies,
        "Contract": contract,
        "PaperlessBilling": paperless_billing,
        "PaymentMethod": payment_method,
        "MonthlyCharges": monthly_charges,
        "TotalCharges": total_charges,
    }

    input_df = pd.DataFrame([input_dict])

    try:
        result = pipeline.predict(input_df)
        pred_label = result["prediction"].iloc[0]
        pred_proba = result["churn_proba"].iloc[0]

        st.subheader("Prediction Result")

        if pred_label == "Yes":
            st.error(
                f"This customer is **LIKELY TO CHURN** "
                f"(churn probability: {pred_proba:.2f})"
            )
        else:
            st.success(
                f"This customer is **NOT LIKELY TO CHURN** "
                f"(churn probability: {pred_proba:.2f})"
            )

        st.caption("Probability is the model's estimated risk that Churn = 'Yes'.")

    except Exception as e:
        st.error(f"Prediction failed: {e}")
