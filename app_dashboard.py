import streamlit as st
import pandas as pd
import os
import numpy as np
import pickle
import io
from typing import Dict, Any, Tuple, List
from datetime import datetime

try:
    from ydata_profiling import ProfileReport
    from streamlit_pandas_profiling import st_profile_report
    HAS_PROFILING = True
except Exception:
    ProfileReport = st_profile_report = None
    HAS_PROFILING = False

from ml_core import ml_pipeline, save_model
import plotly.express as px
import plotly.graph_objects as go

# ── Constants ────────────────────────────────────────────────────────────────

SIZE_PRESETS = {
    "Quick EDA / Notebook visuals":      {"px": (800,  500)},
    "Detailed Comparison Charts":        {"px": (1000, 600)},
    "Correlation Heatmaps / Pairplots":  {"px": (1200, 800)},
    "Presentation / Report visuals":     {"px": (1400, 800)},
    "Multiple subplots (2x2)":           {"px": (1200, 1000)},
}
DATA_PATH = os.path.join("data", "sample_data.csv")
os.makedirs("data", exist_ok=True)

# ── Page config & CSS ────────────────────────────────────────────────────────

st.set_page_config(page_title="🤖 AutoML Pipeline Pro", page_icon="🤖",
                   layout="wide", initial_sidebar_state="expanded")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap');
.main { font-family: 'Inter', sans-serif; }
.main-header {
    font-size:3rem; font-weight:700; text-align:center; margin-bottom:2rem;
    background:linear-gradient(135deg,#667eea,#764ba2);
    -webkit-background-clip:text; -webkit-text-fill-color:transparent;
}
.sec-hdr {
    font-size:1.5rem; font-weight:600; color:#2d3748;
    border-bottom:3px solid #667eea; padding-bottom:.5rem; margin:1.5rem 0 1rem;
}
.card {
    background:linear-gradient(135deg,#667eea,#764ba2);
    padding:1.5rem; border-radius:12px; color:white;
    box-shadow:0 4px 6px rgba(0,0,0,.1);
}
.info-box {
    background:linear-gradient(135deg,#e0f2fe,#bae6fd); color:#0c4a6e;
    padding:1.5rem; border-radius:12px; border-left:4px solid #0ea5e9; margin:1rem 0;
}
.warn-box {
    background:#fff3cd; color:#856404;
    padding:1rem; border-radius:8px; border-left:4px solid #ffc107; margin:1rem 0;
}
.stButton>button { border-radius:8px; font-weight:600; }
</style>
""", unsafe_allow_html=True)

# ── Helpers ──────────────────────────────────────────────────────────────────

@st.cache_data(show_spinner=False)
def load_dataset():
    return pd.read_csv(DATA_PATH, index_col=None) if os.path.exists(DATA_PATH) else None

@st.cache_data(show_spinner=False)
def get_stats(df):
    return dict(rows=df.shape[0], cols=df.shape[1],
                missing=int(df.isnull().sum().sum()),
                memory=df.memory_usage(deep=True).sum()/1024,
                numeric=len(df.select_dtypes(include=[np.number]).columns),
                categorical=len(df.select_dtypes(include=['object']).columns))

def render_chart(df, chart_type, columns, preset):
    w, h = SIZE_PRESETS.get(preset, SIZE_PRESETS["Quick EDA / Notebook visuals"])["px"]
    fig = img_bytes = html_bytes = warn = None
    cs = px.colors.sequential.Viridis

    try:
        if chart_type in ('Bar Chart', 'Column Chart'):
            fig = px.bar(df, x=columns[0], y=columns[1] if len(columns)>1 else df.index,
                         color_discrete_sequence=cs) if columns else None
            warn = warn or (None if columns else "Select at least one column.")

        elif chart_type == 'Line Chart':
            if columns:
                fig = px.line(df, x=columns[0], y=columns[1] if len(columns)>1 else None,
                              line_shape='spline')
            else:
                warn = "Select at least one column."

        elif chart_type == 'Histogram':
            fig = px.histogram(df, x=columns[0], color_discrete_sequence=cs,
                               marginal='box') if columns else None
            warn = warn or (None if columns else "Select one column.")

        elif chart_type in ('Pie Chart', 'Donut Chart'):
            if columns:
                counts = df[columns[0]].value_counts()
                fig = px.pie(names=counts.index, values=counts.values,
                             hole=0.4 if chart_type=='Donut Chart' else 0,
                             color_discrete_sequence=px.colors.qualitative.Set3)
            else:
                warn = "Select a categorical column."

        elif chart_type == 'KPI Chart':
            import matplotlib.pyplot as plt
            vals = [df[c].sum() if pd.api.types.is_numeric_dtype(df[c]) else df[c].count()
                    for c in columns]
            kf, ax = plt.subplots(figsize=(w/100, max(3, h/250)))
            bars = ax.bar(columns, vals, color='#667eea', alpha=0.8)
            for bar, v in zip(bars, vals):
                ax.text(bar.get_x()+bar.get_width()/2, bar.get_height(),
                        f'{v:,.0f}', ha='center', va='bottom', fontweight='bold')
            ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
            plt.xticks(rotation=45, ha='right'); plt.tight_layout()
            buf = io.BytesIO(); kf.savefig(buf, format='png', dpi=150, bbox_inches='tight')
            buf.seek(0); img_bytes = buf.getvalue(); plt.close(kf)
            return None, img_bytes, None, None

        elif chart_type == 'Heatmap Chart':
            if len(columns) >= 2:
                fig = px.density_heatmap(df, x=columns[0], y=columns[1],
                                         color_continuous_scale='Viridis')
            else:
                warn = "Heatmap needs at least two columns."

        if fig is not None:
            fig.update_layout(width=w, height=h, template='plotly_white',
                              font=dict(family='Inter', size=12),
                              margin=dict(l=50, r=50, t=50, b=50))
            html_bytes = fig.to_html().encode('utf-8')

    except Exception as ex:
        warn = f"Error generating chart: {ex}"

    return fig, img_bytes, html_bytes, warn

def metric_card(label, value):
    st.markdown(f"""
    <div class="card" style="margin:.5rem 0">
        <div style="font-size:.9rem;opacity:.9;text-transform:uppercase;letter-spacing:1px">{label}</div>
        <div style="font-size:2.5rem;font-weight:700;margin:.5rem 0">{value}</div>
    </div>""", unsafe_allow_html=True)

def no_dataset_warning():
    st.markdown('<div class="warn-box"><h3 style="margin:0">⚠️ No Dataset Loaded</h3>'
                '<p style="margin:.5rem 0 0">Upload a dataset in the <strong>Upload Dataset</strong> section first.</p>'
                '</div>', unsafe_allow_html=True)

# ── Sidebar ───────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("## 🤖 AutoML Pipeline Pro\n---")
    choice = st.radio("**Navigation**",
                      ["📁 Upload Dataset", "📊 Data Analysis",
                       "🧠 Machine Learning", "💾 Download Models"])
    st.markdown("---")
    df_sb = load_dataset()
    if df_sb is not None:
        s = get_stats(df_sb)
        st.markdown("### 📋 Dataset Overview")
        st.markdown(f"""<div style='background:white;padding:1rem;border-radius:8px;
            box-shadow:0 2px 4px rgba(0,0,0,.1)'>
            <p style='margin:.3rem 0'><b>Rows:</b> {s['rows']:,}</p>
            <p style='margin:.3rem 0'><b>Columns:</b> {s['cols']}</p>
            <p style='margin:.3rem 0'><b>Missing:</b> {s['missing']:,}</p>
            <p style='margin:.3rem 0'><b>Size:</b> {s['memory']:.1f} KB</p>
        </div>""", unsafe_allow_html=True)
    st.markdown("---")
    st.markdown('<div class="card"><p style="margin:0;font-size:.9rem">🚀 <strong>AutoML Pipeline Pro</strong><br>'
                'Advanced ML automation with intelligent preprocessing</p></div>', unsafe_allow_html=True)

df = load_dataset()

# ── Upload Dataset ────────────────────────────────────────────────────────────

if choice == "📁 Upload Dataset":
    st.markdown('<h1 class="main-header">📁 Upload Your Dataset</h1>', unsafe_allow_html=True)
    col1, col2 = st.columns([2, 1])

    with col1:
        st.markdown('<p class="sec-hdr">📂 File Upload</p>', unsafe_allow_html=True)
        file = st.file_uploader("Choose a CSV or Excel file", type=["csv", "xlsx"],
                                help="Max size: 200MB")
        if file:
            try:
                pb = st.progress(0); status = st.empty()
                status.text("📥 Reading file..."); pb.progress(25)
                df = pd.read_csv(file) if file.name.endswith(".csv") else pd.read_excel(file)
                status.text("💾 Saving..."); pb.progress(60)
                df.to_csv(DATA_PATH, index=None)
                st.cache_data.clear(); pb.progress(100); status.text("✅ Done!")

                st.markdown(f'<div class="card"><h3 style="margin:0 0 .5rem">✅ Uploaded {file.name}</h3>'
                            '<p style="margin:0">Ready for analysis</p></div>', unsafe_allow_html=True)

                st.markdown('<p class="sec-hdr">📋 Dataset Preview</p>', unsafe_allow_html=True)
                c1, c2, c3, c4 = st.columns(4)
                for col_w, lbl, val in zip([c1,c2,c3,c4],
                    ["Total Rows","Columns","Missing Values","Numeric Cols"],
                    [f"{df.shape[0]:,}", df.shape[1],
                     df.isnull().sum().sum(),
                     len(df.select_dtypes(include=[np.number]).columns)]):
                    with col_w: metric_card(lbl, val)

                st.dataframe(df.head(10), use_container_width=True, height=400)
            except Exception as e:
                st.error(f"❌ Error: {e}")

    with col2:
        st.markdown('<p class="sec-hdr">📝 Guidelines</p>', unsafe_allow_html=True)
        st.markdown("""<div class="info-box">
            <h4 style="margin-top:0">✅ Supported Formats</h4>
            <ul><li>CSV (.csv)</li><li>Excel (.xlsx)</li></ul>
            <h4>📋 Requirements</h4>
            <ul><li>First row = headers</li><li>No empty columns</li><li>Max 200MB</li></ul>
            <h4>🔧 Auto-Processing</h4>
            <ul style="margin-bottom:0"><li>Missing value detection</li>
            <li>Data type inference</li><li>Memory optimization</li></ul>
        </div>""", unsafe_allow_html=True)

# ── Data Analysis ─────────────────────────────────────────────────────────────

elif choice == "📊 Data Analysis":
    st.markdown('<h1 class="main-header">📊 Exploratory Data Analysis</h1>', unsafe_allow_html=True)

    if df is not None:
        with st.expander("⚙️ Chart Configuration", expanded=True):
            c1, c2, c3 = st.columns(3)
            chart_type  = c1.selectbox("📊 Chart Type",
                ['Bar Chart','Line Chart','Histogram','Pie Chart','KPI Chart','Donut Chart','Heatmap Chart'])
            size_preset = c2.selectbox("📏 Size Preset", list(SIZE_PRESETS.keys()))
            cols_sel    = c3.multiselect("📋 Columns", df.columns,
                default=list(df.columns)[:2] if len(df.columns)>=2 else list(df.columns)[:1])

        tab1, tab2, tab3, tab4 = st.tabs(["📈 Overview","🔍 Custom Charts","📊 Data Quality","📑 Report"])

        with tab1:
            st.markdown('<p class="sec-hdr">📊 Dataset Overview</p>', unsafe_allow_html=True)
            c1, c2, c3 = st.columns(3)
            c1.markdown("#### 🏷️ Data Types")
            c1.dataframe(df.dtypes.value_counts().to_frame("Count"), use_container_width=True)
            c2.markdown("#### 📊 Stats")
            c2.dataframe(df.describe().round(2), use_container_width=True)
            c3.markdown("#### 👁️ Sample")
            c3.dataframe(df.head(5), use_container_width=True)

        with tab2:
            st.markdown('<p class="sec-hdr">📊 Interactive Visualization</p>', unsafe_allow_html=True)
            if cols_sel:
                with st.spinner("🎨 Generating..."):
                    fig, img_bytes, html_bytes, warn = render_chart(df, chart_type, cols_sel, size_preset)
                if warn:
                    st.warning(f"⚠️ {warn}")
                elif fig:
                    st.plotly_chart(fig, use_container_width=True)
                    c1, c2 = st.columns(2)
                    if html_bytes:
                        c1.download_button("💾 Download HTML", html_bytes,
                            f"{chart_type.replace(' ','_')}.html", "text/html", use_container_width=True)
                elif img_bytes:
                    st.image(img_bytes, use_column_width=True)
                if img_bytes:
                    st.download_button("💾 Download PNG", img_bytes,
                        f"{chart_type.replace(' ','_')}.png", "image/png")
            else:
                st.info("👆 Select columns above to generate a chart")

        with tab3:
            st.markdown('<p class="sec-hdr">🔍 Data Quality</p>', unsafe_allow_html=True)
            miss = df.isnull().sum()
            miss = miss[miss > 0].sort_values(ascending=False)
            if len(miss):
                fig_m = go.Figure(go.Bar(x=miss.index, y=miss.values,
                    marker_color='#667eea', text=miss.values, textposition='auto'))
                fig_m.update_layout(title="Missing Values by Column", template='plotly_white', height=400)
                st.plotly_chart(fig_m, use_container_width=True)
                st.dataframe(pd.DataFrame({
                    'Column': miss.index, 'Missing Count': miss.values,
                    'Missing %': (miss.values/len(df)*100).round(2)
                }).reset_index(drop=True), use_container_width=True)
            else:
                st.markdown('<div class="card"><h3 style="margin:0">🎉 No missing values!</h3></div>',
                            unsafe_allow_html=True)

        with tab4:
            st.markdown('<p class="sec-hdr">📑 Profiling Report</p>', unsafe_allow_html=True)
            st.markdown('<div class="warn-box">⏱️ May take several minutes for large datasets.</div>',
                        unsafe_allow_html=True)
            if not HAS_PROFILING:
                st.info("Install `ydata-profiling` and `streamlit-pandas-profiling` to enable.")
            elif st.button("🔄 Generate Report", type="primary"):
                with st.spinner("Analyzing..."):
                    try:
                        st_profile_report(ProfileReport(df, explorative=True, minimal=False,
                                                        title="Dataset Profiling Report"))
                    except Exception as e:
                        st.error(f"❌ {e}")
    else:
        no_dataset_warning()

# ── Machine Learning ──────────────────────────────────────────────────────────

elif choice == "🧠 Machine Learning":
    st.markdown('<h1 class="main-header">🧠 Automated Machine Learning</h1>', unsafe_allow_html=True)

    if df is not None:
        st.markdown('<p class="sec-hdr">⚙️ Model Configuration</p>', unsafe_allow_html=True)
        c1, c2 = st.columns(2)
        target_col = c1.selectbox("🎯 Target Column", df.columns)

        auto_type = "Classification" if df[target_col].dtype=='object' or df[target_col].nunique()<10 else "Regression"
        c2.markdown(f"**Detected:** {'🏷️' if auto_type=='Classification' else '📈'} {auto_type}")
        problem_type = c2.radio("📋 Problem Type", ["Classification","Regression"],
                                index=0 if auto_type=="Classification" else 1)

        st.markdown('<p class="sec-hdr">🤖 Model Selection</p>', unsafe_allow_html=True)
        if problem_type == "Classification":
            avail = ["Logistic Regression","Random Forest","SVM"]
            dflt  = ["Logistic Regression","Random Forest"]
        else:
            avail = ["Linear Regression","Random Forest Regressor","SVR"]
            dflt  = ["Linear Regression","Random Forest Regressor"]

        selected_models = st.multiselect("Select models to train", avail, default=dflt)

        with st.expander("🔧 Advanced Configuration"):
            c1, c2, c3 = st.columns(3)
            test_size    = c1.slider("Test Set Size", 0.1, 0.4, 0.2, 0.05)
            cv_folds     = c2.slider("CV Folds", 3, 10, 5)
            random_state = c3.number_input("Random State", 0, 1000, 42)

        if selected_models:
            if st.button("🚀 Train Models", type="primary", use_container_width=True):
                pb = st.progress(0); status = st.empty()
                status.text("🔄 Preprocessing..."); pb.progress(20)
                with st.spinner("🧠 Training..."):
                    try:
                        results = ml_pipeline(df, target_col, problem_type, selected_models)
                        pb.progress(100); status.text("✅ Done!")
                    except Exception as e:
                        st.error(f"❌ {e}"); results = None

                if results:
                    st.markdown('<p class="sec-hdr">📊 Results</p>', unsafe_allow_html=True)
                    display = {m: {k:v for k,v in metrics.items() if k not in ('Report','Model','Scaler')}
                               for m, metrics in results.items() if 'error' not in metrics}

                    if display:
                        st.dataframe(pd.DataFrame(display).T.style.highlight_max(
                            axis=0, props='background-color:#d4edda;font-weight:bold'),
                            use_container_width=True)

                        if problem_type == "Classification":
                            best = max(results, key=lambda x: results[x].get('Accuracy', 0))
                            score, metric = results[best]['Accuracy'], "Accuracy"
                        else:
                            best = max(results, key=lambda x: results[x].get('R2', -float('inf')))
                            score, metric = results[best]['R2'], "R² Score"

                        st.markdown(f"""<div class="card">
                            <h3 style="margin:0 0 1rem">🏆 Best Model</h3>
                            <h2 style="margin:0;font-size:2rem">{best}</h2>
                            <p style="margin:.5rem 0 0"><strong>{metric}:</strong> {score:.4f}</p>
                        </div>""", unsafe_allow_html=True)

                        st.session_state.update(ml_results=results, best_model=best, problem_type=problem_type)

                        # Comparison chart
                        if len(results) > 1:
                            st.markdown('<p class="sec-hdr">📈 Model Comparison</p>', unsafe_allow_html=True)
                            models = list(results.keys())
                            key1 = 'Accuracy' if problem_type=='Classification' else 'R2'
                            lbl1 = 'Accuracy' if problem_type=='Classification' else 'R² Score'
                            fig = go.Figure([
                                go.Bar(name=lbl1, x=models,
                                       y=[results[m][key1] for m in models], marker_color='#667eea'),
                                go.Bar(name='CV Mean', x=models,
                                       y=[results[m]['CV_Mean'] for m in models], marker_color='#764ba2')
                            ])
                            fig.update_layout(barmode='group', template='plotly_white', height=400)
                            st.plotly_chart(fig, use_container_width=True)

                        st.markdown('<div class="info-box"><h4 style="margin-top:0">✅ Next Steps</h4>'
                                    '<ul style="margin:0"><li>Review metrics above</li>'
                                    '<li>Go to <strong>Download Models</strong> to export</li></ul>'
                                    '</div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="warn-box">⚠️ Select at least one model to train.</div>',
                        unsafe_allow_html=True)
    else:
        no_dataset_warning()

# ── Download Models ───────────────────────────────────────────────────────────

elif choice == "💾 Download Models":
    st.markdown('<h1 class="main-header">💾 Download Trained Models</h1>', unsafe_allow_html=True)

    if st.session_state.get('ml_results'):
        results  = st.session_state['ml_results']
        best     = st.session_state.get('best_model', list(results.keys())[0])
        ptype    = st.session_state['problem_type']

        st.markdown('<p class="sec-hdr">📦 Available Models</p>', unsafe_allow_html=True)

        for model_name, metrics in results.items():
            if 'error' in metrics: continue
            c1, c2, c3 = st.columns([3, 1, 1])

            with c1:
                is_best = model_name == best
                border  = 'background:linear-gradient(135deg,#667eea,#764ba2);color:white' if is_best \
                          else 'background:white;border:2px solid #e2e8f0;color:#2d3748'
                icon    = '🏆' if is_best else '🤖'
                st.markdown(f'<div style="{border};padding:1.5rem;border-radius:12px;margin:.5rem 0">'
                            f'<h3 style="margin:0">{icon} {model_name}</h3>'
                            f'<p style="margin:.3rem 0 0;opacity:.8">{ptype}</p></div>',
                            unsafe_allow_html=True)
                mc1, mc2, mc3 = st.columns(3)
                if ptype == "Classification":
                    mc1.metric("Accuracy", f"{metrics['Accuracy']:.4f}")
                    mc2.metric("CV Mean",  f"{metrics['CV_Mean']:.4f}")
                    mc3.metric("CV Std",   f"{metrics['CV_Std']:.4f}")
                else:
                    mc1.metric("R²",      f"{metrics['R2']:.4f}")
                    mc2.metric("RMSE",    f"{metrics['RMSE']:.4f}")
                    mc3.metric("CV Mean", f"{metrics['CV_Mean']:.4f}")

            with c2:
                try:
                    data = save_model(metrics['Model'], metrics.get('Scaler'), model_name)
                    st.download_button("📥 Download", data,
                        f"{model_name.replace(' ','_').lower()}_model.pkl",
                        "application/octet-stream", key=f"dl_{model_name}", use_container_width=True)
                except Exception as e:
                    st.error(str(e))

            with c3:
                if is_best:
                    st.markdown('<div style="background:#fbbf24;color:white;padding:.5rem;'
                                'border-radius:8px;text-align:center;font-weight:bold">⭐ BEST</div>',
                                unsafe_allow_html=True)
            st.markdown("---")

        # Bulk download
        st.markdown('<p class="sec-hdr">📦 Bulk Download</p>', unsafe_allow_html=True)
        c1, c2 = st.columns([2, 1])
        c1.markdown('<div class="info-box"><h4 style="margin-top:0">💡 Download All Models</h4>'
                    '<p style="margin:0">Get all trained models, scalers, and metrics in one file.</p>'
                    '</div>', unsafe_allow_html=True)
        with c2:
            if st.button("📦 Package All Models", type="primary", use_container_width=True):
                try:
                    pkg = {m: {'model': v['Model'], 'scaler': v.get('Scaler'),
                               'metrics': {k:val for k,val in v.items() if k not in ('Model','Scaler','Report')}}
                           for m, v in results.items() if 'error' not in v}
                    buf = io.BytesIO(); pickle.dump(pkg, buf); buf.seek(0)
                    st.download_button("📥 Download Package", buf.getvalue(),
                        f"automl_models_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pkl",
                        "application/octet-stream", use_container_width=True)
                except Exception as e:
                    st.error(str(e))

        # Usage guide
        st.markdown('<p class="sec-hdr">📖 Usage Guide</p>', unsafe_allow_html=True)
        tab1, tab2 = st.tabs(["🐍 Python Code", "📝 Instructions"])

        with tab1:
            st.code("""import pickle, pandas as pd

with open('model_file.pkl', 'rb') as f:
    data = pickle.load(f)

model, scaler = data['model'], data['scaler']
X_new = pd.DataFrame({'feature1': [v1], 'feature2': [v2]})

if scaler:
    X_new = scaler.transform(X_new)

print(model.predict(X_new))
""", language="python")

        with tab2:
            st.markdown("""
### 🚀 Quick Start
1. **Download** individual models or use **Package All**
2. Load with `pickle.load(open('model.pkl','rb'))`
3. Tree-based models need no scaling; linear models do
4. Match feature order exactly as in training
5. Handle missing values before prediction
            """)

    else:
        st.markdown('<div class="warn-box"><h3 style="margin:0">🔄 No Trained Models</h3>'
                    '<p style="margin:.5rem 0 0">Train models in the <strong>Machine Learning</strong> section first.</p>'
                    '</div>', unsafe_allow_html=True)
        st.markdown('<div class="info-box"><h4 style="margin-top:0">💡 How to Get Started</h4>'
                    '<ol style="margin:0"><li>Upload your dataset</li>'
                    '<li>Explore in Data Analysis</li>'
                    '<li>Train in Machine Learning</li>'
                    '<li>Return here to download</li></ol></div>', unsafe_allow_html=True)

# ── Footer ────────────────────────────────────────────────────────────────────

st.markdown("---")
st.markdown('<div style="text-align:center;color:#718096;padding:2rem 0">'
            '<p style="margin:0;font-size:.9rem">🤖 <strong>AutoML Pipeline Pro</strong> | '
            'Built with Streamlit & scikit-learn</p>'
            '<p style="margin:.5rem 0 0;font-size:.8rem">Automated Machine Learning for Everyone</p>'
            '</div>', unsafe_allow_html=True)
