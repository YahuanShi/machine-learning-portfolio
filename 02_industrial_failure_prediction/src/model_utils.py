import joblib
from xgboost import XGBClassifier
from sklearn.metrics import roc_auc_score

def train_model(X_train, y_train):
    model = XGBClassifier(use_label_encoder=False, eval_metric='logloss')
    model.fit(X_train, y_train)
    return model

def save_model(model, path):
    joblib.dump(model, path)

def load_model(path):
    return joblib.load(path)

def evaluate_model(model, X_test, y_test):
    probs = model.predict_proba(X_test)[:, 1]
    auc = roc_auc_score(y_test, probs)
    return auc