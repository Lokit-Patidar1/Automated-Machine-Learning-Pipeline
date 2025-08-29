# AutoML Django Site

## Setup

1. Create venv and install requirements

```bash
python -m venv .venv
. .venv/Scripts/activate  # Windows PowerShell: .venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

2. Run migrations and start server

```bash
python manage.py migrate
python manage.py runserver 0.0.0.0:8000
```

3. Use the app

- Open http://localhost:8000
- Upload CSV/XLSX -> EDA -> Train -> Download models

## Project Structure

- core_ml/
  - preprocessing.py
  - models_registry.py
  - machine_learning.py
  - eda.py
  - main.py
- webapp/
  - urls.py, views.py
  - templates/
    - base.html, upload.html, eda.html, train.html
- config/
  - settings.py, urls.py, wsgi.py, asgi.py
- manage.py
