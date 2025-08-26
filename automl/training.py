from typing import Dict, Any, List
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, mean_squared_error, r2_score
from sklearn.model_selection import cross_val_score
import streamlit as st

from .models import get_optimized_models


def train_classification_models(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
    selected_models: List[str]
) -> Dict[str, Dict[str, Any]]:
    """Optimized classification model training with cross-validation."""
    models = get_optimized_models("Classification")
    results: Dict[str, Dict[str, Any]] = {}

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    for model_name in selected_models:
        try:
            model = models[model_name]
            if model_name in ["Logistic Regression", "SVM"]:
                model.fit(X_train_scaled, y_train)
                y_pred = model.predict(X_test_scaled)
                cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=5, scoring='accuracy')
            else:
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
                cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='accuracy')

            acc = accuracy_score(y_test, y_pred)
            report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)

            results[model_name] = {
                "Accuracy": round(acc, 4),
                "CV_Mean": round(cv_scores.mean(), 4),
                "CV_Std": round(cv_scores.std(), 4),
                "Report": report,
                "Model": model,
                "Scaler": scaler if model_name in ["Logistic Regression", "SVM"] else None,
            }
        except Exception as e:
            st.error(f"Error training {model_name}: {str(e)}")
            continue

    return results


def train_regression_models(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
    selected_models: List[str]
) -> Dict[str, Dict[str, Any]]:
    """Optimized regression model training with cross-validation."""
    models = get_optimized_models("Regression")
    results: Dict[str, Dict[str, Any]] = {}

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    for model_name in selected_models:
        try:
            model = models[model_name]
            if model_name in ["Linear Regression", "SVR"]:
                model.fit(X_train_scaled, y_train)
                y_pred = model.predict(X_test_scaled)
                cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=5, scoring='r2')
            else:
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
                cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='r2')

            mse = mean_squared_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)
            rmse = np.sqrt(mse)

            results[model_name] = {
                "MSE": round(mse, 4),
                "RMSE": round(rmse, 4),
                "R2": round(r2, 4),
                "CV_Mean": round(cv_scores.mean(), 4),
                "CV_Std": round(cv_scores.std(), 4),
                "Model": model,
                "Scaler": scaler if model_name in ["Linear Regression", "SVR"] else None,
            }
        except Exception as e:
            st.error(f"Error training {model_name}: {str(e)}")
            continue

    return results
