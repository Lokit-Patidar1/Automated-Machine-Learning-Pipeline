# Automated-Machine-Learning-Pipeline
In this project you can upload your data it will give you the most optimal model of classification and regression

## Project Structure

```
Automated-Machine-Learning-Pipeline/
  app.py                      # Streamlit UI entrypoint
  automl/
    __init__.py              # Public API exports
    preprocessing.py         # Data cleaning and encoding
    models.py                # Model factory/configurations
    training.py              # Training and evaluation utilities
    pipeline.py              # End-to-end pipeline orchestration
    serialization.py         # Save/restore helpers for models
```

## How to Run

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Usage
- **Upload Dataset**: CSV or XLSX; cached to `sourcedata.csv`.
- **Data Analysis**: Quick stats and optional detailed profiling.
- **Machine Learning**: Select target and algorithms; compare metrics.
- **Download Models**: Export individual or all trained models.

## Module Imports in app.py
`app.py` imports and uses the modularized pipeline:

```python
from automl import preprocess_data, ml_pipeline, save_model
```