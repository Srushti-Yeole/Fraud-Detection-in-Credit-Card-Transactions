import streamlit as st
import joblib
import numpy as np
import pandas as pd
from datetime import datetime
import os

# Load model and scaler
@st.cache_resource
def load_model():
    model = joblib.load('models/fraud_detection_model.pkl')
    scaler = joblib.load('models/scaler.pkl')
    return model, scaler

model, scaler = load_model()

# Risk assessment logic
def assess_risk(amount, time_of_day, n_transactions, fraud_probability):
    risk_score = fraud_probability

    if amount > 10000:
        risk_score *= 1.5
    elif amount > 5000:
        risk_score *= 1.2

    if time_of_day < 5 or time_of_day > 23:
        risk_score *= 1.3

    if n_transactions > 15:
        risk_score *= 1.4
    elif n_transactions > 10:
        risk_score *= 1.2

    if risk_score > 0.5:
        return "High"
    elif risk_score > 0.2:
        return "Medium"
    return "Low"

# Streamlit UI
st.title("💳 Credit Card Fraud Detection")
st.write("Enter transaction details to assess fraud probability and risk level.")

amount = st.number_input("Transaction Amount (₹)", min_value=0.0, value=1000.0)
time_of_day = st.slider("Time of Transaction (0–24 hrs)", 0.0, 24.0, 12.0)
n_transactions = st.number_input("Recent Transactions Count", min_value=0, value=5)

if st.button("Predict Fraud"):
    input_df = pd.DataFrame([{
        'amount': amount,
        'time_of_day': time_of_day,
        'n_transactions': n_transactions
    }])

    # Scale and predict
    scaled = scaler.transform(input_df)
    prob = float(model.predict_proba(scaled)[0][1])
    risk = assess_risk(amount, time_of_day, n_transactions, prob)

    # Final decision
    blocked = (
        (risk == "High" and amount > 10000) or
        (risk == "High" and time_of_day < 5 and n_transactions > 15) or
        prob > 0.7
    )

    st.subheader("🔍 Prediction Result")
    st.metric("Fraud Probability", f"{prob:.2%}")
    st.metric("Risk Level", risk)
    st.metric("Transaction Blocked", "Yes" if blocked else "No")

    st.info(f"Prediction made at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

