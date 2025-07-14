import pandas as pd
from sklearn.metrics import classification_report, f1_score, roc_auc_score, accuracy_score
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
    y_pred_prob = predict_proba(model, X)[:, 1]
    
    f1 = f1_score(y, y_pred)
    auc = roc_auc_score(y, y_pred_prob)
    acc = accuracy_score(y, y_pred)
    mean_y = y.mean()
    mean_ypred = y_pred.mean()
    
    cr = classification_report(y, y_pred, output_dict=True)
    cr = pd.DataFrame(cr).transpose()

    return acc, f1, auc, mean_y, mean_ypred, cr