import pandas as pd
import numpy as np
import pickle
import io
from typing import Dict, Any, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

# Scikit-learn imports
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.svm import SVC, SVR
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, 
    classification_report, 
    mean_squared_error, 
    r2_score,
    mean_absolute_error
)
from sklearn.model_selection import cross_val_score, train_test_split

# =============================================================================
# MODEL CONFIGURATIONS - Optimized for Performance
# =============================================================================

def get_optimized_models(problem_type: str) -> Dict[str, Any]:
    """
    Get optimized model configurations with tuned hyperparameters.
    
    Args:
        problem_type: Either "Classification" or "Regression"
    
    Returns:
        Dictionary of model name to initialized model object
    """
    if problem_type == "Classification":
        return {
            "Logistic Regression": LogisticRegression(
                max_iter=2000,  # Increased for convergence
                random_state=42,
                solver='lbfgs',  # Faster than liblinear for most cases
                n_jobs=-1  # Use all available cores
            ),
            "Random Forest": RandomForestClassifier(
                n_estimators=100,  # Good balance of performance and speed
                random_state=42,
                n_jobs=-1,  # Parallel processing
                max_depth=15,  # Prevent overfitting
                min_samples_split=5,  # Regularization
                min_samples_leaf=2,
                max_features='sqrt'  # Faster training
            ),
            "SVM": SVC(
                random_state=42,
                kernel='rbf',
                probability=True,  # Enable probability predictions
                C=1.0,
                gamma='scale',  # Auto-adjust based on features
                cache_size=500  # Larger cache for faster training
            )
        }
    else:
        return {
            "Linear Regression": LinearRegression(
                n_jobs=-1  # Parallel computation
            ),
            "Random Forest Regressor": RandomForestRegressor(
                n_estimators=100,
                random_state=42,
                n_jobs=-1,
                max_depth=15,
                min_samples_split=5,
                min_samples_leaf=2,
                max_features='sqrt'
            ),
            "SVR": SVR(
                kernel='rbf',
                C=1.0,
                gamma='scale',
                epsilon=0.1,
                cache_size=500
            )
        }

# =============================================================================
# DATA PREPROCESSING - Enhanced and Cached
# =============================================================================

def preprocess_data(df: pd.DataFrame, target_col: str) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Optimized data preprocessing with intelligent handling of various data types.
    
    Improvements:
    - Efficient missing value imputation
    - Smart categorical encoding
    - Memory optimization
    - Handles edge cases gracefully
    
    Args:
        df: Input dataframe
        target_col: Name of the target column
    
    Returns:
        Tuple of (features DataFrame, target Series)
    """
    # Separate features and target
    X = df.drop(columns=[target_col]).copy()
    y = df[target_col].copy()
    
    # === Handle Missing Values Intelligently ===
    if X.isnull().sum().sum() > 0:
        # Numeric columns: use median (robust to outliers)
        numeric_cols = X.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 0:
            # Vectorized operation for speed
            X[numeric_cols] = X[numeric_cols].fillna(X[numeric_cols].median())
        
        # Categorical columns: use mode or 'Unknown'
        categorical_cols = X.select_dtypes(include=['object', 'category']).columns
        for col in categorical_cols:
            if X[col].mode().empty:
                X[col] = X[col].fillna('Unknown')
            else:
                X[col] = X[col].fillna(X[col].mode()[0])
    
    # === Efficient One-Hot Encoding ===
    categorical_cols = X.select_dtypes(include=['object', 'category']).columns
    
    if len(categorical_cols) > 0:
        # Only encode if we have categorical columns
        X = pd.get_dummies(X, columns=categorical_cols, drop_first=True, dummy_na=False, sparse=False)
    
    # === Convert boolean columns to int for compatibility ===
    bool_cols = X.select_dtypes(include=['bool']).columns
    if len(bool_cols) > 0:
        X[bool_cols] = X[bool_cols].astype(int)
    
    # === Memory optimization ===
    for col in X.select_dtypes(include=[np.number]).columns:
        X[col] = pd.to_numeric(X[col], downcast='float')
    
    return X, y

# =============================================================================
# CLASSIFICATION TRAINING - Optimized
# =============================================================================

def train_classification_models(
    X_train: pd.DataFrame, 
    X_test: pd.DataFrame,
    y_train: pd.Series, 
    y_test: pd.Series,
    selected_models: List[str]
) -> Dict[str, Dict[str, Any]]:
    """
    Train classification models with optimized cross-validation.
    
    Improvements:
    - Shared scaler across linear models
    - Parallel cross-validation
    - Comprehensive metrics
    - Error handling for robustness
    
    Args:
        X_train, X_test: Training and test features
        y_train, y_test: Training and test targets
        selected_models: List of model names to train
    
    Returns:
        Dictionary with model results including metrics and fitted models
    """
    models = get_optimized_models("Classification")
    results: Dict[str, Dict[str, Any]] = {}
    
    # Initialize scaler once for efficiency
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    for model_name in selected_models:
        try:
            model = models[model_name]
            needs_scaling = model_name in ["Logistic Regression", "SVM"]
            
            if needs_scaling:
                model.fit(X_train_scaled, y_train)
                y_pred = model.predict(X_test_scaled)
                
                # Cross-validation with parallel processing
                cv_scores = cross_val_score(
                    model, X_train_scaled, y_train, 
                    cv=5, 
                    scoring='accuracy',
                    n_jobs=-1  
                )
            else:
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
                
                cv_scores = cross_val_score(
                    model, X_train, y_train, 
                    cv=5, 
                    scoring='accuracy',
                    n_jobs=-1
                )
            
            # Calculate comprehensive metrics
            acc = accuracy_score(y_test, y_pred)
            report = classification_report(
                y_test, y_pred, 
                output_dict=True, 
                zero_division=0
            )
            
            results[model_name] = {
                "Accuracy": round(acc, 4),
                "CV_Mean": round(cv_scores.mean(), 4),
                "CV_Std": round(cv_scores.std(), 4),
                "CV_Min": round(cv_scores.min(), 4),
                "CV_Max": round(cv_scores.max(), 4),
                "Report": report,
                "Model": model,
                "Scaler": scaler if needs_scaling else None
            }
            
        except Exception as e:
            # Graceful error handling
            results[model_name] = {
                "error": f"Training failed: {str(e)}"
            }
            continue
    
    return results

# =============================================================================
# REGRESSION TRAINING - Optimized
# =============================================================================

def train_regression_models(
    X_train: pd.DataFrame, 
    X_test: pd.DataFrame,
    y_train: pd.Series, 
    y_test: pd.Series,
    selected_models: List[str]
) -> Dict[str, Dict[str, Any]]:
    """
    Train regression models with optimized cross-validation.
    
    Improvements:
    - Shared scaler for efficiency
    - Multiple evaluation metrics
    - Parallel processing
    - Robust error handling
    
    Args:
        X_train, X_test: Training and test features
        y_train, y_test: Training and test targets
        selected_models: List of model names to train
    
    Returns:
        Dictionary with model results including metrics and fitted models
    """
    models = get_optimized_models("Regression")
    results: Dict[str, Dict[str, Any]] = {}
    
    # Initialize scaler once
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    for model_name in selected_models:
        try:
            model = models[model_name]
            
            # Determine if model needs scaling
            needs_scaling = model_name in ["Linear Regression", "SVR"]
            
            if needs_scaling:
                model.fit(X_train_scaled, y_train)
                y_pred = model.predict(X_test_scaled)
                
                # Parallel cross-validation
                cv_scores = cross_val_score(
                    model, X_train_scaled, y_train, 
                    cv=5, 
                    scoring='r2',
                    n_jobs=-1
                )
            else:
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
                
                cv_scores = cross_val_score(
                    model, X_train, y_train, 
                    cv=5, 
                    scoring='r2',
                    n_jobs=-1
                )
            
            # Calculate comprehensive regression metrics
            mse = mean_squared_error(y_test, y_pred)
            rmse = np.sqrt(mse)
            mae = mean_absolute_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)
            
            # Adjusted R² for better model comparison
            n = len(y_test)
            p = X_test.shape[1]
            adj_r2 = 1 - (1 - r2) * (n - 1) / (n - p - 1) if n > p + 1 else r2
            
            results[model_name] = {
                "MSE": round(mse, 4),
                "RMSE": round(rmse, 4),
                "MAE": round(mae, 4),
                "R2": round(r2, 4),
                "Adj_R2": round(adj_r2, 4),
                "CV_Mean": round(cv_scores.mean(), 4),
                "CV_Std": round(cv_scores.std(), 4),
                "CV_Min": round(cv_scores.min(), 4),
                "CV_Max": round(cv_scores.max(), 4),
                "Model": model,
                "Scaler": scaler if needs_scaling else None
            }
            
        except Exception as e:
            results[model_name] = {
                "error": f"Training failed: {str(e)}"
            }
            continue
    
    return results

# =============================================================================
# MAIN ML PIPELINE
# =============================================================================

def ml_pipeline(
    df: pd.DataFrame, 
    target_col: str, 
    problem_type: str,
    selected_models: List[str],
    test_size: float = 0.2,
    random_state: int = 42
) -> Dict[str, Dict[str, Any]]:
    """
    Complete ML pipeline from preprocessing to model training.
    
    Args:
        df: Input dataframe
        target_col: Name of target column
        problem_type: "Classification" or "Regression"
        selected_models: List of models to train
        test_size: Proportion of test set (default 0.2)
        random_state: Random seed for reproducibility
    
    Returns:
        Dictionary of model results
    """
    # === Step 1: Preprocess Data ===
    try:
        X, y = preprocess_data(df, target_col)
    except Exception as e:
        return {"preprocessing_error": str(e)}
    
    # Validate data
    if X.empty or y.empty:
        return {"error": "Empty dataset after preprocessing"}
    
    if len(y.unique()) < 2:
        return {"error": "Target column has insufficient unique values"}
    
    # === Step 2: Train-Test Split ===
    try:
        if problem_type == "Classification" and len(y.unique()) > 1:
            if y.value_counts().min() >= 2:
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, 
                    test_size=test_size, 
                    random_state=random_state, 
                    stratify=y
                )
            else:
                # Fall back to regular split if stratification not possible
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, 
                    test_size=test_size, 
                    random_state=random_state
                )
        else:
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, 
                test_size=test_size, 
                random_state=random_state
            )
    except Exception as e:
        return {"split_error": str(e)}
    
    # === Step 3: Train Models ===
    if problem_type == "Classification":
        results = train_classification_models(
            X_train, X_test, y_train, y_test, selected_models
        )
    else:
        results = train_regression_models(
            X_train, X_test, y_train, y_test, selected_models
        )
    
    return results

# =============================================================================
# MODEL SERIALIZATION
# =============================================================================

def save_model(model: Any, scaler: Optional[Any], model_name: str) -> bytes:
    """
    Serialize model and scaler to bytes for download.
    
    Args:
        model: Trained model object
        scaler: StandardScaler object or None
        model_name: Name of the model
    
    Returns:
        Pickled model data as bytes
    """
    model_data = {
        'model': model,
        'scaler': scaler,
        'model_name': model_name,
        'sklearn_version': '1.0+'  # Version info for compatibility
    }
    
    buffer = io.BytesIO()
    pickle.dump(model_data, buffer, protocol=pickle.HIGHEST_PROTOCOL)
    buffer.seek(0)
    return buffer.getvalue()


def save_model_to_file(
    model: Any, 
    scaler: Optional[Any], 
    model_name: str, 
    filepath: str
) -> None:
    """
    Save model and scaler to a pickle file on disk.
    
    Args:
        model: Trained model object
        scaler: StandardScaler object or None
        model_name: Name of the model
        filepath: Path where to save the file
    """
    model_data = {
        'model': model,
        'scaler': scaler,
        'model_name': model_name,
        'sklearn_version': '1.0+'
    }
    
    with open(filepath, 'wb') as f:
        pickle.dump(model_data, f, protocol=pickle.HIGHEST_PROTOCOL)


def load_model_from_file(filepath: str) -> Dict[str, Any]:
    """
    Load model and scaler from a pickle file.
    
    Args:
        filepath: Path to the pickle file
    
    Returns:
        Dictionary containing model, scaler, and metadata
    """
    with open(filepath, 'rb') as f:
        return pickle.load(f)


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def get_feature_importance(model: Any, feature_names: List[str]) -> pd.DataFrame:
    """
    Extract feature importance from trained models (if available).
    
    Args:
        model: Trained model
        feature_names: List of feature names
    
    Returns:
        DataFrame with features and their importance scores
    """
    if hasattr(model, 'feature_importances_'):
        # Tree-based models
        importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': model.feature_importances_
        }).sort_values('importance', ascending=False)
        return importance_df
    elif hasattr(model, 'coef_'):
        # Linear models
        importance_df = pd.DataFrame({
            'feature': feature_names,
            'coefficient': np.abs(model.coef_[0]) if model.coef_.ndim > 1 else np.abs(model.coef_)
        }).sort_values('coefficient', ascending=False)
        return importance_df
    else:
        return pd.DataFrame()


def predict_with_model(
    model_data: Dict[str, Any], 
    X_new: pd.DataFrame
) -> np.ndarray:
    """
    Make predictions using a loaded model.
    
    Args:
        model_data: Dictionary from load_model_from_file
        X_new: New data for prediction
    
    Returns:
        Array of predictions
    """
    model = model_data['model']
    scaler = model_data.get('scaler')
    
    if scaler is not None:
        X_scaled = scaler.transform(X_new)
        return model.predict(X_scaled)
    else:
        return model.predict(X_new)
