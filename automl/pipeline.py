from typing import Dict, Any, List
import pandas as pd
import streamlit as st
from sklearn.model_selection import train_test_split

from .preprocessing import preprocess_data
from .training import train_classification_models, train_regression_models


def ml_pipeline(
    df: pd.DataFrame,
    target_col: str,
    problem_type: str,
    selected_models: List[str]
) -> Dict[str, Dict[str, Any]]:
    """Optimized ML pipeline with error handling and preprocessing."""
    try:
        X, y = preprocess_data(df, target_col)

        if X.empty or y.empty:
            st.error("Dataset is empty after preprocessing")
            return {}

        if len(X) < 10:
            st.warning("Dataset is very small. Results may not be reliable.")

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
    except Exception as e:
        st.error(f"Error in ML pipeline: {str(e)}")
        return {}
