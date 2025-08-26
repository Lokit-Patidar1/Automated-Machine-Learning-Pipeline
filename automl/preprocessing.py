import pandas as pd
import numpy as np
from typing import Tuple


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


