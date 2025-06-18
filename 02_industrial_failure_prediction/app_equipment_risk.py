
import streamlit as st
import pandas as pd
import joblib
import numpy as np

st.set_page_config(page_title="Early Failure Risk Detection", layout="centered")
st.title("🛠️ Equipment Failure Prediction Dashboard")

# Load model
MODEL_PATH = 'models/early_failure_model.pkl'
model = joblib.load(MODEL_PATH)

st.sidebar.header("🔧 Input Sensor Statistics")

# Input fields (statistical features from sliding window)
input_data = {
    'sensor2_mean': st.sidebar.slider("Sensor2 - Mean", 0.0, 1.5, 0.6),
    'sensor2_std': st.sidebar.slider("Sensor2 - Std", 0.0, 1.0, 0.3),
    'sensor2_diff': st.sidebar.slider("Sensor2 - Diff", -1.0, 1.0, 0.1),
    'sensor3_mean': st.sidebar.slider("Sensor3 - Mean", 0.0, 1.5, 0.8),
    'sensor3_std': st.sidebar.slider("Sensor3 - Std", 0.0, 1.0, 0.2),
    'sensor3_diff': st.sidebar.slider("Sensor3 - Diff", -1.0, 1.0, 0.0),
    'sensor7_mean': st.sidebar.slider("Sensor7 - Mean", 500, 800, 700),
    'sensor7_std': st.sidebar.slider("Sensor7 - Std", 0, 100, 20),
    'sensor7_diff': st.sidebar.slider("Sensor7 - Diff", -100, 100, 10),
    'sensor11_mean': st.sidebar.slider("Sensor11 - Mean", 20, 60, 40),
    'sensor11_std': st.sidebar.slider("Sensor11 - Std", 0, 20, 5),
    'sensor11_diff': st.sidebar.slider("Sensor11 - Diff", -20, 20, 2)
}

df_input = pd.DataFrame([input_data])

# Prediction
if st.button("🔍 Predict Failure Risk"):
    proba = model.predict_proba(df_input)[0, 1]
    st.metric("🔢 Failure Probability", f"{proba:.2%}")
    if proba < 0.3:
        st.success("✅ Low Risk")
    elif proba < 0.7:
        st.warning("⚠️ Medium Risk")
    else:
        st.error("❗ High Risk — Immediate Check Recommended")
