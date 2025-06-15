# evaluate.py
# Improved evaluation script for LightGBM with feature importance plot and classification report

import lightgbm as lgb
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, mean_squared_error, classification_report
import os

# Load dataset
print("Dataset: CSV file loaded via pandas")
df = pd.read_csv('data/high_diamond_ranked_10min.csv') 

# Prepare features and target
drop_cols = ['gameId', 'blueWins']
X = df.drop(drop_cols, axis=1)
y = df['blueWins']

# Load trained model
model = lgb.Booster(model_file='models/lightgbm_model.txt')

# Predict
y_pred = model.predict(X)
# For classification:
y_pred_binary = (y_pred > 0.5).astype(int)

# Evaluate
acc = accuracy_score(y, y_pred_binary)
print(f"Full Dataset Accuracy: {acc:.4f}")

# Print classification report
print("\nClassification Report:")
print(classification_report(y, y_pred_binary))

# Save predictions
os.makedirs('results', exist_ok=True)
results_df = pd.DataFrame({
    'y_true': y,
    'y_pred_proba': y_pred,
    'y_pred': y_pred_binary  # For classification
})

results_df.to_csv('results/lightgbm_predictions.csv', index=False)
print("Predictions saved to results/lightgbm_predictions.csv")

# Save feature importance plot
print("Saving feature importance plot...")
lgb.plot_importance(model)
plt.tight_layout()
plt.savefig('results/lightgbm_feature_importance.png')
print("Feature importance plot saved to results/lightgbm_feature_importance.png")
