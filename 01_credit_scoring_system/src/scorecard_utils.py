import numpy as np
import joblib

def prob_to_score(prob, base_point=600, PDO=50, odds0=1/19):
    """Convert probability to credit score using logistic scaling."""
    prob = np.clip(prob, 1e-6, 1 - 1e-6)
    odds = (1 - prob) / prob
    score = base_point + (np.log(odds / odds0) / np.log(2)) * (-PDO)
    return score

def save_model(model, path="models/scorecard_model.pkl"):
    """Save trained model to disk."""
    joblib.dump(model, path)

def load_model(path="models/scorecard_model.pkl"):
    """Load model from disk."""
    return joblib.load(path)
