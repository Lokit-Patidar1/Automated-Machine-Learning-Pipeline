"""
Consolidated Machine Learning Engine
Contains all ML functionality: preprocessing, models, training, and serialization
"""

import pandas as pd
import numpy as np
import pickle
import io
from typing import Dict, Any, List, Tuple
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.svm import SVC, SVR
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, mean_squared_error, r2_score
from sklearn.model_selection import cross_val_score, train_test_split


def get_optimized_models(problem_type: str) -> Dict[str, Any]:
    """Get optimized model configurations"""
    if problem_type == "Classification":
        return {
            "Logistic Regression": LogisticRegression(
                max_iter=1000,
                random_state=42,
                solver='liblinear'
            ),
            "Random Forest": RandomForestClassifier(
                n_estimators=100,
                random_state=42,
                n_jobs=-1,
                max_depth=10
            ),
            "SVM": SVC(
                random_state=42,
                kernel='rbf',
                probability=True
            )
        }
    else:
        return {
            "Linear Regression": LinearRegression(),
            "Random Forest Regressor": RandomForestRegressor(
                n_estimators=100,
                random_state=42,
                n_jobs=-1,
                max_depth=10
            ),
            "SVR": SVR(
                kernel='rbf',
                C=1.0
            )
        }


def preprocess_data(df: pd.DataFrame, target_col: str) -> Tuple[pd.DataFrame, pd.Series]:
    """Optimized data preprocessing with caching"""
    X = df.drop(columns=[target_col]).copy()
    y = df[target_col].copy()

    # Handle missing values
    if X.isnull().sum().sum() > 0:
        # Fill numeric columns with median
        numeric_cols = X.select_dtypes(include=[np.number]).columns
        X[numeric_cols] = X[numeric_cols].fillna(X[numeric_cols].median())

        # Fill categorical columns with mode
        categorical_cols = X.select_dtypes(include=['object']).columns
        for col in categorical_cols:
            X[col] = X[col].fillna(X[col].mode()[0] if not X[col].mode().empty else 'Unknown')

    # Convert categorical variables with better handling
    X = pd.get_dummies(X, drop_first=True, dummy_na=False)

    return X, y


def train_classification_models(X_train: pd.DataFrame, X_test: pd.DataFrame,
                              y_train: pd.Series, y_test: pd.Series,
                              selected_models: List[str]) -> Dict[str, Dict[str, Any]]:
    """Optimized classification model training with cross-validation"""
    models = get_optimized_models("Classification")
    results: Dict[str, Dict[str, Any]] = {}

    # Scale features for SVM and Logistic Regression
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
                "Scaler": scaler if model_name in ["Logistic Regression", "SVM"] else None
            }

        except Exception as e:
            results[model_name] = {"error": str(e)}
            continue

    return results


def train_regression_models(X_train: pd.DataFrame, X_test: pd.DataFrame,
                          y_train: pd.Series, y_test: pd.Series,
                          selected_models: List[str]) -> Dict[str, Dict[str, Any]]:
    """Optimized regression model training with cross-validation"""
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
                "Scaler": scaler if model_name in ["Linear Regression", "SVR"] else None
            }

        except Exception as e:
            results[model_name] = {"error": str(e)}
            continue

    return results


def ml_pipeline(df: pd.DataFrame, target_col: str, problem_type: str,
               selected_models: List[str]) -> Dict[str, Dict[str, Any]]:
    """Optimized ML pipeline with better error handling and preprocessing"""
    # Preprocess data
    X, y = preprocess_data(df, target_col)

    if X.empty or y.empty:
        return {}

    # Train-test split
    if problem_type == "Classification" and len(y.unique()) > 1:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
    else:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

    if problem_type == "Classification":
        results = train_classification_models(X_train, X_test, y_train, y_test, selected_models)
    else:
        results = train_regression_models(X_train, X_test, y_train, y_test, selected_models)

    return results


def save_model(model: Any, scaler: Any, model_name: str) -> bytes:
    """Save model and scaler to bytes for download using pickle"""
    model_data = {
        'model': model,
        'scaler': scaler,
        'model_name': model_name
    }

    buffer = io.BytesIO()
    pickle.dump(model_data, buffer)
    buffer.seek(0)
    return buffer.getvalue()


def save_model_to_file(model: Any, scaler: Any, model_name: str, filepath: str) -> None:
    """Save model and scaler to a pickle file"""
    model_data = {
        'model': model,
        'scaler': scaler,
        'model_name': model_name
    }
    
    with open(filepath, 'wb') as f:
        pickle.dump(model_data, f)


def load_model_from_file(filepath: str) -> Dict[str, Any]:
    """Load model and scaler from a pickle file"""
    with open(filepath, 'rb') as f:
        return pickle.load(f)
