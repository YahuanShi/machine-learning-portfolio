# Version: v2.0 (Streamlit App for Risk Scoring)
# Features: Manual Input, Batch Upload, Risk Segmentation, Download Support
import streamlit as st
import pandas as pd
import joblib
import numpy as np
import json
import scorecardpy as sc

# Paths
MODEL_PATH = 'models/xgboost_model.pkl'
BIN_PATH = 'outputs/woe_bins.json'

# Load model and bins
model = joblib.load(MODEL_PATH)
with open(BIN_PATH, 'r') as f:
    bins_dict = json.load(f)
bins = {k: pd.DataFrame(v) for k, v in bins_dict.items()}

st.set_page_config(page_title="Credit Scoring App", layout="centered")
st.title("📊 Credit Scoring with Risk Classification")

# Input mode
mode = st.sidebar.radio("Select Mode", ["Manual Input", "Batch Upload"])

def apply_woe(df_raw):
    return sc.woebin_ply(df_raw, bins)

if mode == "Manual Input":
    st.subheader("🧾 Enter Applicant Information")

    input_data = {
        'age': st.number_input("Age", min_value=18, max_value=100, value=35),
        'DebtRatio': st.slider("Debt Ratio", min_value=0.0, max_value=3.0, value=0.5),
        'MonthlyIncome': st.number_input("Monthly Income", min_value=0, value=5000),
        'NumberOfDependents': st.slider("Number of Dependents", 0, 10, 1),
        'NumberOfOpenCreditLinesAndLoans': st.slider("Open Credit Lines", 0, 50, 8),
        'RevolvingUtilizationOfUnsecuredLines': st.slider("Revolving Utilization", 0.0, 2.0, 0.5),
        'NumberOfTime30-59DaysPastDueNotWorse': st.slider("30-59 Days Late", 0, 10, 1),
        'NumberOfTime60-89DaysPastDueNotWorse': st.slider("60-89 Days Late", 0, 10, 0),
        'NumberOfTimes90DaysLate': st.slider("90+ Days Late", 0, 10, 0)
    }

    df_input = pd.DataFrame([input_data])
    df_woe = apply_woe(df_input)

    if st.button("Predict Score"):
        proba = model.predict_proba(df_woe)[:, 1][0]
        st.metric("📈 Default Probability", f"{proba:.2%}")

        if proba < 0.2:
            level = "🟢 Low Risk"
        elif proba < 0.5:
            level = "🟡 Medium Risk"
        elif proba < 0.8:
            level = "🟠 High Risk"
        else:
            level = "🔴 Critical Risk"

        st.success(f"Risk Level: {level}")

else:
    st.subheader("📁 Upload Batch Data (Raw CSV)")
    file = st.file_uploader("Upload CSV file", type="csv")

    if file:
        raw = pd.read_csv(file)
        df_woe = apply_woe(raw)
        preds = model.predict_proba(df_woe)[:, 1]
        raw['default_proba'] = preds
        raw['risk_level'] = pd.cut(preds, [-np.inf, 0.2, 0.5, 0.8, np.inf],
                                   labels=['Low', 'Medium', 'High', 'Critical'])
        st.write(raw.head())
        st.download_button("Download Scored Data", raw.to_csv(index=False), file_name="scored.csv")
