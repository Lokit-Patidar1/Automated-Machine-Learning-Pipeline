# Automated Machine Learning Pipeline

An automated machine learning pipeline that provides an intuitive web interface for data analysis, model training, and model deployment. Upload your dataset and get the most optimal classification and regression models with just a few clicks.

## 🚀 Features

- **Easy Data Upload**: Support for CSV and Excel files
- **Exploratory Data Analysis**: Comprehensive data profiling and visualization
- **Automated ML Pipeline**: Train multiple models and compare performance
- **Model Export**: Download trained models as pickle files
- **User-Friendly Interface**: Clean, modern Streamlit-based UI

## 📁 Project Structure

```
Automated-Machine-Learning-Pipeline/
├── app.py                    # Streamlit web application
├── ml_engine.py             # Consolidated ML engine (all ML functionality)
├── requirements.txt         # Python dependencies
├── sourcedata.csv          # Sample dataset (auto-generated)
└── README.md               # This file
```

## 🚀 Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Run the application
streamlit run app.py
```

The application will open in your browser at `http://localhost:8501`

## 🎯 Usage

1. **Upload Dataset**: Navigate to the "Upload Dataset" section and upload your CSV or Excel file
2. **Data Analysis**: Use the "Data Analysis" section to explore your data with comprehensive statistics and visualizations
3. **Machine Learning**: In the "Machine Learning" section:
   - Select your target column
   - Choose the problem type (Classification or Regression)
   - Select which models to train
   - Click "Train Models" to start the automated pipeline
4. **Download Models**: Export your trained models as pickle files for deployment

## 🤖 Supported Models

### Classification
- Logistic Regression
- Random Forest Classifier
- Support Vector Machine (SVM)

### Regression
- Linear Regression
- Random Forest Regressor
- Support Vector Regression (SVR)

## 📦 Model Export

Trained models are exported as pickle files containing:
- The trained model
- Preprocessing scaler (if applicable)
- Model metadata

### Using Exported Models

```python
import pickle
import pandas as pd

# Load the model
with open('model_file.pkl', 'rb') as f:
    model_data = pickle.load(f)

model = model_data['model']
scaler = model_data['scaler']  # May be None for some models

# Make predictions
if scaler:
    X_scaled = scaler.transform(X_new)
    predictions = model.predict(X_scaled)
else:
    predictions = model.predict(X_new)
```

## 🔧 Dependencies

- **streamlit** (≥1.28.0): Web application framework
- **pandas** (≥1.5.0): Data manipulation and analysis
- **numpy** (2.1.3): Numerical computing
- **matplotlib** (≥3.8.0): Data visualization
- **scikit-learn** (≥1.3.0): Machine learning algorithms
- **ydata-profiling** (≥4.0.0): Data profiling and EDA
- **streamlit-pandas-profiling** (≥0.1.3): Streamlit integration for data profiling
- **openpyxl** (≥3.1.0): Excel file support

## 📊 Features in Detail

### Data Preprocessing
- Automatic missing value handling
- Categorical variable encoding
- Feature scaling for appropriate models
- Train-test split with stratification

### Model Evaluation
- Cross-validation scoring
- Multiple performance metrics
- Model comparison visualizations
- Best model recommendation

### User Interface
- Responsive design
- Real-time progress indicators
- Interactive data exploration
- Download capabilities

## 🤝 Contributing

Feel free to submit issues and enhancement requests!

## 📄 License

This project is open source and available under the MIT License.