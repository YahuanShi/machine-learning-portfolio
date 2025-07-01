
# Multi-Component RUL & Fault Prediction System

This project implements a **multi-task LSTM model** that jointly estimates the **Remaining Useful Life (RUL)** and classifies **component-level faults** for aircraft engines. The project is built on the **NASA C-MAPSS FD004 dataset**, designed to simulate multiple simultaneous degradation modes.

---

## Project Structure

```
03_multi-target_remaining_useful_life_estimator/
├── data/                    # Placeholder for raw CSV files
├── outputs/                 # Numpy files for processed sequences and labels
├── models/                 # Trained multitask LSTM model (multitask_lstm.pt)
├── notebooks/
│   ├── 01_data_preparation.ipynb      # Create sequences + labels
│   └── 02_multitask_training.ipynb    # Train joint RUL + fault predictor
│   └── 03_multitask_evaluation.ipynb
├── src/
│   ├── multitask_model.py             # LSTM model definition
│   └── utils.py                       # Utilities for processing, evaluation
├── app_multitask_rul.py               # Streamlit app for interactive prediction
├── README.md
```

---

## Features

- ✅ Multi-task training: RUL regression + Fault classification
- ✅ Normalized RUL labels (0–1) for better convergence
- ✅ Robust loss function using `SmoothL1Loss` (RUL) and `BCELoss` (Fault)
- ✅ Expanded LSTM capacity (hidden=128, dropout=0.2)
- ✅ Visual Streamlit dashboard for model demo

---

## Model Architecture

A shared LSTM backbone with dual output heads:
- 🔵 RUL Head: Fully connected layer with linear output
- 🔴 Fault Head: Fully connected layer with sigmoid activation

---

## Dataset Used

**NASA C-MAPSS FD004 Subset**

- Multi-component failures (3 fault modes)
- Sensor readings from 21 channels
- Engine life cycles with early fault annotation

---

## How to Run

### 1. Prepare data
Process C-MAPSS FD004 raw file into model-ready format:

```bash
jupyter notebook 01_data_preparation.ipynb
```

### 2. Train model

```bash
jupyter notebook 02_multitask_training.ipynb
```

A model file will be saved at `models/multitask_lstm.pt`.

### 3. Evaluation model

```bash
jupyter notebook 03_multitask_evaluation.ipynb
```

### 4. Launch Streamlit demo

```bash
streamlit run app_multitask_rul.py
```

Interactively adjust input sensor values and get live RUL & fault risk prediction.

---

## Performance Metrics (Val Set)

- RUL RMSE ≈ ~18.2
- Fault AUC ≈ ~0.93
- Stable convergence across 20 epochs

---

## Version History

| Version | Description |
|---------|-------------|
| v1.0    | Baseline multi-task LSTM with MSE loss |
| v2.0    | Improved version: Normalization + SmoothL1 + Better generalization |

---

## Credits

- Dataset: [NASA C-MAPSS](https://www.nasa.gov/cmapps)
- Model inspired by deep multitask architecture for industrial equipment monitoring
