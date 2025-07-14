import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, f1_score
from src.config import RANDOM_STATE
from src.modeling.predict import evaluate_model, predict_class


def split_data(X, y, test_size=0.2, random_state=RANDOM_STATE):
    """divide data train/test"""
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    print(f"Train: {X_train.shape}, Test: {X_test.shape}")
    return X_train, X_test, y_train, y_test


def train_model(X_train, y_train, model_params=None):
    """Entrena modelo RandomForest"""
    if model_params is None:
        model_params = {
            'n_estimators': 100,
            'max_depth': 10,
            'random_state': RANDOM_STATE,
            'n_jobs': -1
        }
    
    print(f"Training RandomForest with params: {model_params}")
    model = RandomForestClassifier(**model_params)
    model.fit(X_train, y_train)
    print("Training completed")
    
    return model

def save_model(model, filepath):
    """Guarda modelo"""
    joblib.dump(model, filepath)
    print(f"Model saved to: {filepath}")


def train_and_save(X, y, model_path, model_params=None, test_size=0.2):
    """Complete pipeline: split -> train -> evaluate -> save"""
    print("Starting training pipeline...")
    
    # Split data
    X_train, X_test, y_train, y_test = split_data(X, y, test_size)
    
    # Train model
    model = train_model(X_train, y_train, model_params)
    
    # Evaluate
    acc, f1, auc, _, _, cr = evaluate_model(model, X_test, y_test)

    print(f"\nAccuracy: {acc:.3f}")
    print(f"F1-Score: {f1:.3f}")
    print(f"AUC: {auc:.3f}")
    print(f"Classification Report:\n{cr}")
    
    # Save
    save_model(model, model_path)
    
    print(f"Pipeline completed.")
    return model
