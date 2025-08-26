from typing import Dict, Any
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.svm import SVC, SVR


def get_optimized_models(problem_type: str) -> Dict[str, Any]:
    """Get optimized model configurations by problem type."""
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
