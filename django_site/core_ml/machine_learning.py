from typing import Dict, Any, List
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, mean_squared_error, r2_score
from sklearn.model_selection import cross_val_score, train_test_split

from .preprocessing import preprocess
from .models_registry import get_models


def train_and_evaluate(df: pd.DataFrame, target_col: str, problem_type: str, selected_models: List[str]) -> Dict[str, Dict[str, Any]]:
	X, y = preprocess(df, target_col)
	if X.empty or y.empty:
		return {}
	if problem_type == 'Classification' and len(y.unique()) > 1:
		X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
	else:
		X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
	models = get_models(problem_type)
	results: Dict[str, Dict[str, Any]] = {}
	# single scaler reused per model-family
	scaler = StandardScaler()
	X_train_scaled = scaler.fit_transform(X_train)
	X_test_scaled = scaler.transform(X_test)
	for model_name in selected_models:
		try:
			model = models[model_name]
			if problem_type == 'Classification':
				if model_name in ['Logistic Regression', 'SVM']:
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
					'Accuracy': round(acc, 4),
					'CV_Mean': round(cv_scores.mean(), 4),
					'CV_Std': round(cv_scores.std(), 4),
					'Report': report,
					'Model': model,
					'Scaler': scaler if model_name in ['Logistic Regression', 'SVM'] else None,
				}
			else:
				if model_name in ['Linear Regression', 'SVR']:
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
					'MSE': round(mse, 4),
					'RMSE': round(rmse, 4),
					'R2': round(r2, 4),
					'CV_Mean': round(cv_scores.mean(), 4),
					'CV_Std': round(cv_scores.std(), 4),
					'Model': model,
					'Scaler': scaler if model_name in ['Linear Regression', 'SVR'] else None,
				}
		except Exception as e:
			results[model_name] = {'error': str(e)}
			continue
	return results
