import streamlit as st
import pandas as pd
import os

st.set_page_config(page_title="AutoMLpipeline", layout="wide")

# Note: Heavy ML libraries (scikit-learn) are imported lazily inside functions

# ----------------- CACHING HELPERS ----------------- #
@st.cache_data(show_spinner=False)
def load_dataframe_from_disk(path: str):
    return pd.read_csv(path, index_col=None)

@st.cache_data(show_spinner=False)
def read_uploaded_file(file):
    if file.name.endswith(".csv"):
        return pd.read_csv(file)
    return pd.read_excel(file)

@st.cache_data(show_spinner=True, max_entries=5)
def cached_ml_pipeline(df, target_col, problem_type, selected_models):
    # Ensure stable hashing of selection for cache key
    if isinstance(selected_models, list):
        selected_models = tuple(selected_models)
    return ml_pipeline(df, target_col, problem_type, selected_models)

# ----------------- ML FUNCTIONS ----------------- #
def train_classification_models(X_train, X_test, y_train, y_test, selected_models):
    # Lazy-import heavy estimators
    from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.svm import SVC
    from sklearn.metrics import accuracy_score, classification_report
    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000),
        "Random Forest": RandomForestClassifier(n_jobs=-1),
        "SVM": SVC()
    }
    results = {}

    for model_name in selected_models:
        model = models[model_name]
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        report = classification_report(y_test, y_pred, output_dict=True)
        results[model_name] = {"Accuracy": acc, "Report": report}
    return results


def train_regression_models(X_train, X_test, y_train, y_test, selected_models):
    # Lazy-import heavy estimators
    from sklearn.linear_model import LinearRegression
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.svm import SVR
    from sklearn.metrics import mean_squared_error, r2_score
    models = {
        "Linear Regression": LinearRegression(),
        "Random Forest Regressor": RandomForestRegressor(n_jobs=-1),
        "SVR": SVR()
    }
    results = {}

    for model_name in selected_models:
        model = models[model_name]
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        results[model_name] = {"MSE": mse, "R2": r2}
    return results


def ml_pipeline(df, target_col, problem_type, selected_models):
    from sklearn.model_selection import train_test_split
    X = df.drop(columns=[target_col])
    y = df[target_col]

    # Convert categorical variables
    X = pd.get_dummies(X, drop_first=True)

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Run based on problem type
    if problem_type == "Classification":
        results = train_classification_models(X_train, X_test, y_train, y_test, selected_models)
    else:
        results = train_regression_models(X_train, X_test, y_train, y_test, selected_models)

    return results
# ------------------------------------------------ #

# ----------------- STREAMLIT APP ---------------- #
with st.sidebar:
    st.title("AutoMLpipeline")
    choice = st.radio("Navigation", ["Upload Your Dataset", "Exploratory Data Analysis", "ML", "Download"])
    st.info("This application allows you to build an automated ML pipeline using Streamlit")


# Load dataset if exists (cached)
df = None
if os.path.exists("sourcedata.csv"):
    df = load_dataframe_from_disk("sourcedata.csv")


if choice == "Upload Your Dataset":
    st.title("Upload Your Dataset Here")
    file = st.file_uploader("Upload your file (CSV/Excel)", type=["csv", "xlsx"])
    if file:
        df = read_uploaded_file(file)
        df.to_csv("sourcedata.csv", index=None)
        st.dataframe(df.head())


elif choice == "Exploratory Data Analysis":
    st.title("Automated Exploratory Data Analysis")
    if df is not None:
        st.subheader("Dataset Overview")
        st.write(df.shape)
        st.write(df.dtypes)
        st.write(df.describe())
        # Heavy EDA profiling gated behind a button and lazy imports
        enable_minimal = st.checkbox("Use minimal profiling (faster, fewer computations)", value=True)
        suggested_rows = min(5000, len(df)) if len(df) > 0 else 0
        max_rows = st.number_input("Max rows to profile (0 = all rows)", min_value=0, value=suggested_rows)
        if st.button("Generate Profiling Report"):
            # Optionally sample to speed up profiling
            df_to_profile = df if max_rows == 0 else df.head(int(max_rows))
            try:
                from ydata_profiling import ProfileReport
                from streamlit_pandas_profiling import st_profile_report
            except Exception:
                st.warning("Profiling dependencies are not installed. Install 'ydata-profiling' and 'streamlit-pandas-profiling'.")
            else:
                with st.spinner("Generating profile report..."):
                    profile_report = ProfileReport(
                        df_to_profile,
                        explorative=not enable_minimal,
                        minimal=enable_minimal,
                    )
                    st_profile_report(profile_report)
    else:
        st.warning("Please upload a dataset first.")


elif choice == "ML":
    st.title("Automated Machine Learning")

    if df is not None:
        target_col = st.selectbox("Select Target Column", df.columns)
        problem_type = st.radio("Select Problem Type", ["Classification", "Regression"])

        if problem_type == "Classification":
            available_models = ["Logistic Regression", "Random Forest", "SVM"]
        else:
            available_models = ["Linear Regression", "Random Forest Regressor", "SVR"]

        selected_models = st.multiselect("Select Models to Train", available_models, default=available_models)

        if st.button("Train Models"):
            # Cache full ML run to speed up repeated experiments
            results = cached_ml_pipeline(df, target_col, problem_type, selected_models)

            # Show results
            st.subheader("Model Performance")
            st.write(pd.DataFrame(results).T)

            best_model_choice = st.selectbox("Select Best Model", results.keys())
            st.success(f"You selected: {best_model_choice}")
    else:
        st.warning("Please upload a dataset first.")


elif choice == "Download":
    st.title("Download Section")
    st.info("You can implement model or report download functionality here.")