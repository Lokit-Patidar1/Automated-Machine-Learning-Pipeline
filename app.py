# Import files 
import streamlit as st
import pandas as pd
import os 
import numpy as np
from typing import Dict, Any, Tuple, List
import pickle
import io

# EDA import files  
from ydata_profiling import ProfileReport
from streamlit_pandas_profiling import st_profile_report

# ML imports 
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, mean_squared_error, r2_score
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.svm import SVC, SVR
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import cross_val_score
import warnings
warnings.filterwarnings('ignore')

# ----------------- ML FUNCTIONS ----------------- #
@st.cache_data
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

def train_classification_models(X_train: pd.DataFrame, X_test: pd.DataFrame, 
                              y_train: pd.Series, y_test: pd.Series, 
                              selected_models: List[str]) -> Dict[str, Dict[str, Any]]:
    """Optimized classification model training with cross-validation"""
    models = get_optimized_models("Classification")
    results = {}
    
    # Scale features for SVM and Logistic Regression
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    for model_name in selected_models:
        try:
            model = models[model_name]
            
            # Use scaled data for models that benefit from scaling
            if model_name in ["Logistic Regression", "SVM"]:
                model.fit(X_train_scaled, y_train)
                y_pred = model.predict(X_test_scaled)
                # Cross-validation score
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
            st.error(f"Error training {model_name}: {str(e)}")
            continue
    
    return results

def train_regression_models(X_train: pd.DataFrame, X_test: pd.DataFrame, 
                          y_train: pd.Series, y_test: pd.Series, 
                          selected_models: List[str]) -> Dict[str, Dict[str, Any]]:
    """Optimized regression model training with cross-validation"""
    models = get_optimized_models("Regression")
    results = {}
    
    # Scale features for SVR and Linear Regression
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    for model_name in selected_models:
        try:
            model = models[model_name]
            
            # Use scaled data for models that benefit from scaling
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
            st.error(f"Error training {model_name}: {str(e)}")
            continue
    
    return results

def ml_pipeline(df: pd.DataFrame, target_col: str, problem_type: str, 
               selected_models: List[str]) -> Dict[str, Dict[str, Any]]:
    """Optimized ML pipeline with better error handling and preprocessing"""
    try:
        # Preprocess data
        X, y = preprocess_data(df, target_col)
        
        # Validate data
        if X.empty or y.empty:
            st.error("Dataset is empty after preprocessing")
            return {}
        
        if len(X) < 10:
            st.warning("Dataset is very small. Results may not be reliable.")
        
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
        
    except Exception as e:
        st.error(f"Error in ML pipeline: {str(e)}")
        return {}

def save_model(model, scaler, model_name: str) -> bytes:
    """Save model and scaler to bytes for download"""
    model_data = {
        'model': model,
        'scaler': scaler,
        'model_name': model_name
    }
    
    buffer = io.BytesIO()
    pickle.dump(model_data, buffer)
    buffer.seek(0)
    return buffer.getvalue()
# ------------------------------------------------ #

# ----------------- STREAMLIT APP ---------------- #
# Page configuration
st.set_page_config(
    page_title="AutoML Pipeline",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .success-message {
        background-color: #d4edda;
        color: #155724;
        padding: 1rem;
        border-radius: 0.5rem;
        border: 1px solid #c3e6cb;
    }
</style>
""", unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.markdown("# 🤖 AutoML Pipeline")
    choice = st.radio(
        "Navigation", 
        ["📁 Upload Dataset", "📊 Data Analysis", "🧠 Machine Learning", "💾 Download Models"],
        help="Navigate through different sections of the AutoML pipeline"
    )
    
    st.markdown("---")
    st.info("🚀 This application provides an automated machine learning pipeline with advanced preprocessing, model training, and evaluation capabilities.")
    
    # Dataset info in sidebar
    if os.path.exists("sourcedata.csv"):
        df_info = pd.read_csv("sourcedata.csv", index_col=None)
        st.markdown("### 📋 Current Dataset")
        st.write(f"**Shape:** {df_info.shape[0]} rows × {df_info.shape[1]} columns")
        st.write(f"**Size:** {df_info.memory_usage(deep=True).sum() / 1024:.1f} KB")

# Load dataset with caching
@st.cache_data
def load_dataset():
    if os.path.exists("sourcedata.csv"):
        return pd.read_csv("sourcedata.csv", index_col=None)
    return None

df = load_dataset()

# Main content
if choice == "📁 Upload Dataset":
    st.markdown('<h1 class="main-header">📁 Upload Your Dataset</h1>', unsafe_allow_html=True)
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("### Upload File")
        file = st.file_uploader(
            "Choose a CSV or Excel file", 
            type=["csv", "xlsx"],
            help="Upload your dataset in CSV or Excel format. The file will be automatically processed and cached."
        )
        
        if file:
            try:
                with st.spinner("Processing file..."):
                    if file.name.endswith(".csv"):
                        df = pd.read_csv(file)
                    else:
                        df = pd.read_excel(file)
                    
                    # Save to cache
                    df.to_csv("sourcedata.csv", index=None)
                    st.cache_data.clear()  # Clear cache to reload data
                    
                st.success(f"✅ Successfully uploaded {file.name}")
                
                # Display dataset preview
                st.markdown("### 📋 Dataset Preview")
                st.dataframe(df.head(10), use_container_width=True)
                
                # Dataset statistics
                col1_stats, col2_stats, col3_stats = st.columns(3)
                with col1_stats:
                    st.metric("Total Rows", df.shape[0])
                with col2_stats:
                    st.metric("Total Columns", df.shape[1])
                with col3_stats:
                    st.metric("Missing Values", df.isnull().sum().sum())
                    
            except Exception as e:
                st.error(f"❌ Error processing file: {str(e)}")
    
    with col2:
        st.markdown("### 📝 Instructions")
        st.markdown("""
        1. **Supported formats:** CSV, Excel (.xlsx)
        2. **File requirements:**
           - First row should contain column headers
           - No completely empty columns
           - Reasonable file size (< 200MB)
        3. **Data types:** Numeric and categorical data supported
        4. **Missing values:** Will be handled automatically
        """)

elif choice == "📊 Data Analysis":
    st.markdown('<h1 class="main-header">📊 Exploratory Data Analysis</h1>', unsafe_allow_html=True)
    
    if df is not None:
        # Quick stats
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Rows", df.shape[0])
        with col2:
            st.metric("Columns", df.shape[1])
        with col3:
            st.metric("Numeric Columns", len(df.select_dtypes(include=[np.number]).columns))
        with col4:
            st.metric("Categorical Columns", len(df.select_dtypes(include=['object']).columns))
        
        # Tabs for different analyses
        tab1, tab2, tab3 = st.tabs(["📈 Basic Info", "🔍 Detailed Analysis", "📊 Data Quality"])
        
        with tab1:
            col1, col2 , col3 = st.columns(3)
            with col1:
                st.markdown("### Data Types")
                st.dataframe(df.dtypes.to_frame('Data Type'), use_container_width=True)
            
            with col2:
                st.markdown("### Statistical Summary")
                st.dataframe(df.describe(), use_container_width=True)

            with col3:
                st.markdown("### First 5 rows")
                st.dataframe(df.head(5), use_container_width=True)
        
        with tab2:
            if st.button("🔄 Generate Detailed Report", help="This may take a few moments for large datasets"):
                with st.spinner("Generating comprehensive analysis report..."):
                    try:
                        profile_report = ProfileReport(df, explorative=True, minimal=True)
                        st_profile_report(profile_report)
                    except Exception as e:
                        st.error(f"Error generating report: {str(e)}")
                        st.info("Try with a smaller dataset or check data quality.")
        
        with tab3:
            st.markdown("### Missing Values Analysis")
            missing_data = df.isnull().sum()
            missing_data = missing_data[missing_data > 0].sort_values(ascending=False)
            
            if len(missing_data) > 0:
                st.bar_chart(missing_data)
                st.dataframe(
                    pd.DataFrame({
                        'Column': missing_data.index,
                        'Missing Count': missing_data.values,
                        'Missing %': (missing_data.values / len(df) * 100).round(2)
                    }).reset_index(drop=True),
                    use_container_width=True
                )
            else:
                st.success("🎉 No missing values found in the dataset!")
    else:
        st.warning("⚠️ Please upload a dataset first in the Upload Dataset section.")

elif choice == "🧠 Machine Learning":
    st.markdown('<h1 class="main-header">🧠 Automated Machine Learning</h1>', unsafe_allow_html=True)

    if df is not None:
        # Configuration section
        st.markdown("### ⚙️ Model Configuration")
        col1, col2 = st.columns(2)
        
        with col1:
            target_col = st.selectbox(
                "🎯 Select Target Column", 
                df.columns,
                help="Choose the column you want to predict"
            )
            
        with col2:
            # Auto-detect problem type based on target column
            if df[target_col].dtype == 'object' or df[target_col].nunique() < 10:
                default_type = "Classification"
            else:
                default_type = "Regression"
                
            problem_type = st.radio(
                "📋 Problem Type", 
                ["Classification", "Regression"],
                index=0 if default_type == "Classification" else 1,
                help="Classification for categorical targets, Regression for continuous targets"
            )

        # Model selection
        if problem_type == "Classification":
            available_models = ["Logistic Regression", "Random Forest", "SVM"]
            default_models = ["Logistic Regression", "Random Forest"]
        else:
            available_models = ["Linear Regression", "Random Forest Regressor", "SVR"]
            default_models = ["Linear Regression", "Random Forest Regressor"]

        selected_models = st.multiselect(
            "🤖 Select Models to Train", 
            available_models, 
            default=default_models,
            help="Choose which algorithms to train and compare"
        )

        # Advanced options
        with st.expander("🔧 Advanced Options"):
            test_size = st.slider("Test Set Size", 0.1, 0.4, 0.2, 0.05)
            cv_folds = st.slider("Cross-Validation Folds", 3, 10, 5)
            random_state = st.number_input("Random State", 0, 1000, 42)

        # Training section
        if selected_models:
            if st.button("🚀 Train Models", type="primary"):
                with st.spinner("Training models... This may take a few moments."):
                    results = ml_pipeline(df, target_col, problem_type, selected_models)

                if results:
                    st.markdown("### 📊 Model Performance Results")
                    
                    # Create results dataframe for display
                    display_results = {}
                    for model_name, metrics in results.items():
                        display_results[model_name] = {
                            k: v for k, v in metrics.items() 
                            if k not in ['Report', 'Model', 'Scaler']
                        }
                    
                    results_df = pd.DataFrame(display_results).T
                    st.dataframe(results_df, use_container_width=True)
                    
                    # Best model recommendation
                    if problem_type == "Classification":
                        best_model = max(results.keys(), key=lambda x: results[x]['Accuracy'])
                        best_score = results[best_model]['Accuracy']
                        metric_name = "Accuracy"
                    else:
                        best_model = max(results.keys(), key=lambda x: results[x]['R2'])
                        best_score = results[best_model]['R2']
                        metric_name = "R² Score"
                    
                    st.markdown(f"""
                    <div class="success-message">
                        <h4>🏆 Best Model: {best_model}</h4>
                        <p><strong>{metric_name}:</strong> {best_score:.4f}</p>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Store results in session state for download
                    st.session_state['ml_results'] = results
                    st.session_state['best_model'] = best_model
                    st.session_state['problem_type'] = problem_type
                    
                    # Model comparison chart
                    if len(results) > 1:
                        st.markdown("### 📈 Model Comparison")
                        if problem_type == "Classification":
                            chart_data = pd.DataFrame({
                                'Model': list(results.keys()),
                                'Accuracy': [results[model]['Accuracy'] for model in results.keys()],
                                'CV Mean': [results[model]['CV_Mean'] for model in results.keys()]
                            })
                            st.bar_chart(chart_data.set_index('Model'))
                        else:
                            chart_data = pd.DataFrame({
                                'Model': list(results.keys()),
                                'R² Score': [results[model]['R2'] for model in results.keys()],
                                'CV Mean': [results[model]['CV_Mean'] for model in results.keys()]
                            })
                            st.bar_chart(chart_data.set_index('Model'))
        else:
            st.warning("⚠️ Please select at least one model to train.")
    else:
        st.warning("⚠️ Please upload a dataset first in the Upload Dataset section.")

elif choice == "💾 Download Models":
    st.markdown('<h1 class="main-header">💾 Download Trained Models</h1>', unsafe_allow_html=True)
    
    if 'ml_results' in st.session_state and st.session_state['ml_results']:
        results = st.session_state['ml_results']
        best_model_name = st.session_state.get('best_model', list(results.keys())[0])
        
        st.markdown("### 📋 Available Models")
        
        for model_name in results.keys():
            col1, col2, col3 = st.columns([2, 1, 1])
            
            with col1:
                is_best = model_name == best_model_name
                badge = " 🏆" if is_best else ""
                st.write(f"**{model_name}{badge}**")
                
                if st.session_state['problem_type'] == "Classification":
                    st.write(f"Accuracy: {results[model_name]['Accuracy']:.4f}")
                else:
                    st.write(f"R² Score: {results[model_name]['R2']:.4f}")
            
            with col2:
                # Download individual model
                model_data = save_model(
                    results[model_name]['Model'],
                    results[model_name].get('Scaler'),
                    model_name
                )
                st.download_button(
                    label="📥 Download",
                    data=model_data,
                    file_name=f"{model_name.replace(' ', '_').lower()}_model.pkl",
                    mime="application/octet-stream",
                    key=f"download_{model_name}"
                )
            
            with col3:
                if is_best:
                    st.success("Best Model")
        
        # Download all models
        st.markdown("---")
        if st.button("📦 Download All Models"):
            all_models_data = {}
            for model_name in results.keys():
                all_models_data[model_name] = {
                    'model': results[model_name]['Model'],
                    'scaler': results[model_name].get('Scaler'),
                    'metrics': {k: v for k, v in results[model_name].items() 
                              if k not in ['Model', 'Scaler', 'Report']}
                }
            
            buffer = io.BytesIO()
            pickle.dump(all_models_data, buffer)
            buffer.seek(0)
            
            st.download_button(
                label="📥 Download All Models Package",
                data=buffer.getvalue(),
                file_name="automl_models_package.pkl",
                mime="application/octet-stream"
            )
        
        # Usage instructions
        with st.expander("📖 How to Use Downloaded Models"):
            st.markdown("""
            ```python
            import pickle
            import pandas as pd
            
            # Load the model
            with open('model_file.pkl', 'rb') as f:
                model_data = pickle.load(f)
            
            model = model_data['model']
            scaler = model_data['scaler']  # May be None for some models
            
            # Make predictions
            # If scaler exists, scale your data first
            if scaler:
                X_scaled = scaler.transform(X_new)
                predictions = model.predict(X_scaled)
            else:
                predictions = model.predict(X_new)
            ```
            """)
    else:
        st.info("🔄 No trained models available. Please train some models in the Machine Learning section first.")
