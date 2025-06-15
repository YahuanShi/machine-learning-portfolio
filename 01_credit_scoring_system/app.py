
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns


import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "src")))
from scorecard_utils import prob_to_score, load_model

st.set_page_config(page_title="Credit Scoring App", layout="wide")
st.title("📊 Credit Scoring System (Logistic Model)")

# Load model
model = load_model("models/logistic_model.pkl")

# Sidebar mode selection
mode = st.sidebar.radio("Select Mode", ["🔍 Single Prediction", "📂 Batch Scoring"])
base_point = st.sidebar.number_input("Base Score", value=600)
pdo = st.sidebar.number_input("Points to Double Odds (PDO)", value=50)
odds0 = st.sidebar.number_input("Base Odds (bad:good)", value=1/19)

# Risk level categorization
def risk_level(score):
    if score > 720:
        return "🟢 Low"
    elif score > 620:
        return "🟡 Medium"
    else:
        return "🔴 High"

# Single Prediction Mode
if mode == "🔍 Single Prediction":
    st.subheader("🧮 Manual Input")
    with st.form("input_form"):
        age = st.number_input("Age", min_value=18, max_value=100, value=35)
        debt_ratio = st.slider("Debt Ratio", 0.0, 2.0, 0.3)
        revol_util = st.slider("Revolving Utilization", 0.0, 2.0, 0.6)
        open_credit_lines = st.number_input("Open Credit Lines", 0, 30, 10)
        past_due = st.number_input("Num Past Due >30 Days", 0, 10, 0)
        submitted = st.form_submit_button("Score")

    if submitted:
        df_input = pd.DataFrame([{
            "age": age,
            "DebtRatio": debt_ratio,
            "RevolvingUtilizationOfUnsecuredLines": revol_util,
            "NumberOfOpenCreditLinesAndLoans": open_credit_lines,
            "NumberOfTime30-59DaysPastDueNotWorse": past_due
        }])
        prob = model.predict_proba(df_input)[0, 1]
        score = prob_to_score(prob, base_point, pdo, odds0)
        st.success(f"Predicted Probability of Default: {prob:.2%}")
        st.info(f"Credit Score: {score:.0f}")
        st.warning(f"Risk Level: {risk_level(score)}")

# Batch Scoring Mode
elif mode == "📂 Batch Scoring":
    st.subheader("📁 Upload CSV File")
    uploaded_file = st.file_uploader("Upload your CSV file", type="csv")

    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        if 'SeriousDlqin2yrs' in df.columns:
            df = df.drop(columns=['SeriousDlqin2yrs'])

        # Score
        prob = model.predict_proba(df)[:, 1]
        df['prob'] = prob
        df['score'] = prob_to_score(prob, base_point, pdo, odds0)
        df['risk'] = df['score'].apply(risk_level)

        st.dataframe(df.head())

        # Plot score distribution
        st.subheader("📈 Score Distribution")
        fig, ax = plt.subplots()
        sns.histplot(df['score'], kde=True, bins=20, ax=ax)
        st.pyplot(fig)

        # Download results
        st.subheader("📤 Download Scored Results")
        csv = df.to_csv(index=False).encode("utf-8")
        st.download_button("Download CSV", csv, "scored_results.csv", "text/csv")
