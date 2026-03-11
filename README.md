# Automated Machine Learning Pipeline

A Streamlit app for quick EDA, model training, and exporting trained models. It supports classification and regression with common baselines and provides charts, profiling, and download utilities.

## Project Structure
- `app_dashboard.py` - Streamlit UI
- `ml_core.py` - Training pipeline and model utilities
- `deps.txt` - Python dependencies
- `data/sample_data.csv` - Sample dataset (optional)

## Setup
1. Create and activate a virtual environment (recommended).
2. Install dependencies:

```bash
pip install -r deps.txt
```

## Run
```bash
streamlit run app_dashboard.py
```

## Notes
- Upload your own CSV or Excel file from the UI. Uploaded datasets are saved to `data/sample_data.csv`.
- Model artifacts are downloaded through the UI.
