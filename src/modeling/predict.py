import pandas as pd
from sklearn.metrics import classification_report, f1_score
import joblib
from src.config import RANDOM_STATE

def load_model(filepath):
    """Load model using joblib"""
    return joblib.load(filepath)

def predict_class(model, X):
    return model.predict(X)

def predict_proba(model, X):
    return model.predict_proba(X)

def evaluate_model(model, X, y):
    """Evaluate model and return metrics"""
    y_pred = predict_class(model, X)
    f1 = f1_score(y, y_pred)
    cr = classification_report(y, y_pred, output_dict=True)
    cr = pd.DataFrame(cr).transpose()

    print(f"F1-Score: {f1:.3f}")
    return f1, cr