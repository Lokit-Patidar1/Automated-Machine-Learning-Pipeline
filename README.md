# Automated Machine Learning Pipeline

Production-ready Streamlit ML app with a GitHub Pages landing site.

## Stack Detected
- Python + Streamlit (`app.py`)
- Scikit-learn ML engine (`ml_engine.py`)
- Static GitHub Pages landing page (`index.html`, `styles/styles.css`, `scripts/main.js`)

## Clean Project Structure

```text
Automated-Machine-Learning-Pipeline/
├── .github/
│   └── workflows/
│       └── deploy-pages.yml
├── .streamlit/
│   └── config.toml
├── assets/
│   └── .gitkeep
├── data/
│   └── sample.csv
├── scripts/
│   └── main.js
├── styles/
│   └── styles.css
├── .gitignore
├── .nojekyll
├── app.py
├── index.html
├── ml_engine.py
├── README.md
└── requirements.txt
```

## Local Run

```bash
pip install -r requirements.txt
streamlit run app.py
```

Expected URL: `http://localhost:8501`

## GitHub Deployment

### 1. Initialize and commit
```bash
git init
git add .
git commit -m "Initial clean deployment"
```

### 2. Push to GitHub
```bash
git branch -M main
git remote add origin <MY_GITHUB_REPO_URL>
git push -u origin main
```

### 3. Enable GitHub Pages
1. Open repository on GitHub.
2. Go to `Settings -> Pages`.
3. Under `Build and deployment`, choose `Source: GitHub Actions`.
4. Keep workflow file `.github/workflows/deploy-pages.yml` on `main`.
5. Push to `main`; GitHub will publish your static landing page.

## Important Deployment Note

GitHub Pages hosts only static files.  
Your full ML app (`app.py`) must be deployed on Streamlit Community Cloud:

1. Go to `https://share.streamlit.io/`
2. Select your GitHub repo and `app.py`
3. Deploy
4. Put your Streamlit app URL in `index.html` (Deploy button)
