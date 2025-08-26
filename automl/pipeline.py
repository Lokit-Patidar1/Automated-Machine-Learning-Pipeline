from typing import Dict, Any, List
import pandas as pd
from sklearn.model_selection import train_test_split

from .preprocessing import preprocess_data
from .training import train_classification_models, train_regression_models


def ml_pipeline(df: pd.DataFrame, target_col: str, problem_type: str,
               selected_models: List[str]) -> Dict[str, Dict[str, Any]]:
    """Optimized ML pipeline with better error handling and preprocessing"""
    # Preprocess data
    X, y = preprocess_data(df, target_col)

    if X.empty or y.empty:
        return {}

    # Train-test split with stratification for classification
    if problem_type == "Classification" and len(y.unique()) > 1:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
    else:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

    # Run based on problem type
    if problem_type == "Classification":
        results = train_classification_models(X_train, X_test, y_train, y_test, selected_models)
    else:
        results = train_regression_models(X_train, X_test, y_train, y_test, selected_models)

    return results


