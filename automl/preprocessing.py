import pandas as pd
import numpy as np
from typing import Tuple


def preprocess_data(df: pd.DataFrame, target_col: str) -> Tuple[pd.DataFrame, pd.Series]:
    """Optimized data preprocessing with caching handled in app.
    - Splits features/target
    - Imputes missing values (median for numeric, mode for categorical)
    - One-hot encodes categoricals with drop_first
    """
    X = df.drop(columns=[target_col]).copy()
    y = df[target_col].copy()

    if X.isnull().sum().sum() > 0:
        numeric_cols = X.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 0:
            X[numeric_cols] = X[numeric_cols].fillna(X[numeric_cols].median())

        categorical_cols = X.select_dtypes(include=['object']).columns
        for col in categorical_cols:
            mode_series = X[col].mode()
            X[col] = X[col].fillna(mode_series[0] if not mode_series.empty else 'Unknown')

    X = pd.get_dummies(X, drop_first=True, dummy_na=False)
    return X, y
