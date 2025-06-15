# Credit Scoring System

This project builds a complete credit scoring system based on the [Give Me Some Credit](https://www.kaggle.com/c/GiveMeSomeCredit) dataset. It includes data cleaning, WOE binning, IV analysis, scorecard modeling (logistic regression & XGBoost), score transformation, and a Streamlit web demo for real-world simulation.

## Structure

```
.
├── app.py                      # Streamlit Web App
├── data/
│   └── cs-training.csv         # Raw input data (Kaggle)
├── models/
│   └── logistic_model.pkl      # Trained model file
├── notebooks/
│   └── 01_woe_binning_modeling.ipynb  # Full modeling notebook
├── outputs/                    # Score distributions, exported results
├── src/
│   └── scorecard_utils.py      # Utility functions (score mapping, save/load model)
└── requirements.txt
```

---

## Key Features

- End-to-end credit scorecard modeling pipeline
- WOE binning + IV-based feature selection
- Logistic regression and XGBoost model training
- Score mapping (probability → score)
- Risk level classification
- Streamlit demo (manual & batch scoring)
- Exportable prediction results

---

## Methods Overview

| Step | Description |
|------|-------------|
| Data | Clean and prepare Kaggle dataset |
| Feature Eng. | WOE binning using `scorecardpy` |
| Model | Logistic Regression and XGBoost |
| Evaluation | AUC, ROC curve, score distribution plots |
| Score Mapping | Probability → Log odds → Score (Base=600, PDO=50) |
| Web App | Built with Streamlit: single + batch prediction |


---

## Web App Usage

> Launch the demo locally:

```bash
streamlit run app.py
```

### Manual Scoring  
- Input individual features  
- View probability, score, risk level

### Batch Scoring  
- Upload CSV file with feature columns  
- Get scored output + download predictions  
- See score distribution chart

---

Dependencies

Install all requirements:

```bash
pip install -r requirements.txt
```
---

## To Improve

- Add SHAP feature importance
- Support more models (LightGBM, CatBoost)
- Build REST API version (Flask/FastAPI)
- Integrate with real-time scoring interface

---

## License

This project is for learning and demonstration purposes. You are free to use and adapt it for non-commercial work.
