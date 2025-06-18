# Industrial Equipment Failure Prediction

A full machine learning pipeline to predict early-stage failures in turbofan engines using NASA's CMAPSS dataset.

---

## Project Structure

```
industrial-failure-prediction/
├── data/                         # Raw & processed CMAPSS data
├── outputs/                      # Engineered feature CSVs
├── models/                       # Trained models (.pkl)
├── notebooks/                    # Step-by-step Jupyter notebooks
├── src/                    # Core function script
├── app_equipment_risk.py         # Streamlit dashboard
├── requirements.txt              # Environment dependencies
└── README.md
```

---

## Features

- NASA CMAPSS Dataset
- Time-series feature engineering (rolling mean, std, diff)
- Binary classification of early failure (RUL ≤ 30)
- XGBoost model + SHAP explainability
- Streamlit UI dashboard for real-time risk scoring

---

## Notebooks

| File | Description |
|------|-------------|
| `01_data_exploration.ipynb` | Load & explore raw cycles + sensors |
| `02_feature_engineering.ipynb` | Rolling window stats + label creation |
| `03_modeling_ui_features_only.ipynb` | Train model using only UI fields |
| `04_app_equipment_risk.py` | Streamlit UI for risk prediction |

---

## How to Use

```bash
pip install -r requirements.txt
streamlit run app_equipment_risk.py
```

Ensure you have:
- Trained model at `models/early_failure_model.pkl`
- Features saved from previous notebooks

---

## Reference

- CMAPSS Dataset: [NASA Turbofan Engine Degradation](https://data.nasa.gov/d/ff5v-kuh6)