import pandas as pd
import numpy as np
from typing import Tuple


def preprocess(df: pd.DataFrame, target_col: str) -> Tuple[pd.DataFrame, pd.Series]:
	X = df.drop(columns=[target_col]).copy()
	y = df[target_col].copy()
	if X.isnull().sum().sum() > 0:
		numeric_cols = X.select_dtypes(include=[np.number]).columns
		X[numeric_cols] = X[numeric_cols].fillna(X[numeric_cols].median())
		categorical_cols = X.select_dtypes(include=['object']).columns
		for col in categorical_cols:
			mode_val = X[col].mode()[0] if not X[col].mode().empty else 'Unknown'
			X[col] = X[col].fillna(mode_val)
	X = pd.get_dummies(X, drop_first=True, dummy_na=False)
	return X, y
