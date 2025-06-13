# lightgbm_train.py
# Improved training script for LightGBM with reproducibility and parameter saving

import lightgbm as lgb
import pandas as pd
import numpy as np
import random
import os
import json
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, mean_squared_error

# Fix random seed for reproducibility
def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

seed_everything(42)

# Load dataset
print("Dataset: CSV file loaded via pandas")
df = pd.read_csv('data/high_diamond_ranked_10min.csv') 

# Prepare features and target
drop_cols = ['gameId', 'blueWins']
X = df.drop(drop_cols, axis=1)
y = df['blueWins']

# Split into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Create LightGBM dataset
train_data = lgb.Dataset(X_train, label=y_train)
val_data = lgb.Dataset(X_val, label=y_val)

# Define model parameters
params = {
    'objective': 'binary',  # Change to 'regression' or 'multiclass' if needed
    'metric': 'binary_logloss',  # Change metric accordingly
    'verbosity': -1,
    'seed': 42
}

# Save parameters to JSON
os.makedirs('models', exist_ok=True)
with open('models/lightgbm_params.json', 'w') as f:
    json.dump(params, f)
print("Saved training parameters to models/lightgbm_params.json")

# Train model
print("Starting training...")
model = lgb.train(
    params,
    train_data,
    valid_sets=[val_data],
    num_boost_round=100,
    early_stopping_rounds=10
)

# Predict
y_pred = model.predict(X_val)
# For classification:
y_pred_binary = (y_pred > 0.5).astype(int)

# Evaluate
acc = accuracy_score(y_val, y_pred_binary)
print(f"Validation Accuracy: {acc:.4f}")

# Save model
model.save_model('models/lightgbm_model.txt')
print("Training complete. Model saved to models/lightgbm_model.txt")
