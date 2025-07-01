import streamlit as st
import torch
import numpy as np
import matplotlib.pyplot as plt
from src.multitask_model import MultiTaskLSTM

st.set_page_config(layout="wide")
st.title("🔧 Multi-Component RUL + Fault Classifier")

# Load model and data
X = np.load('outputs/X.npy')
idx = st.slider("Select sample index", 0, len(X)-1, 0)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = MultiTaskLSTM(input_dim=X.shape[2])
model.load_state_dict(torch.load('models/multitask_lstm.pt', map_location=device))
model.to(device).eval()

with torch.no_grad():
    x_input = torch.tensor(X[idx:idx+1], dtype=torch.float32).to(device)
    rul_pred, fault_pred = model(x_input)
    rul_pred = rul_pred.cpu().numpy().flatten()
    fault_pred = fault_pred.cpu().numpy().flatten()

# Display results
st.subheader("🔢 Predicted RUL")
cols = st.columns(3)
for i, comp in enumerate(['comp1', 'comp2', 'comp3']):
    cols[i].metric(label=comp, value=f"{rul_pred[i]:.1f} cycles")

st.subheader("⚠️ Fault Probabilities")
fig, ax = plt.subplots()
ax.bar(['comp1', 'comp2', 'comp3'], fault_pred, color='tomato')
ax.set_ylim(0, 1)
ax.set_ylabel("Probability")
st.pyplot(fig)

st.caption("Use the slider above to view different test samples")
