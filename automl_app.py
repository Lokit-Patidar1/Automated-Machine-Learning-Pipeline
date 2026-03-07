import streamlit as st
import pandas as pd
import os 
import numpy as np
from typing import Dict, Any, Tuple, List
import pickle
import io

# EDA import files (optional)
try:
    from ydata_profiling import ProfileReport
    from streamlit_pandas_profiling import st_profile_report
    HAS_PROFILING = True
except Exception:
    ProfileReport = None
    st_profile_report = None
    HAS_PROFILING = False

# Internal ML Engine
from training_engine import (
    ml_pipeline,
    save_model,
)

import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt
from datetime import datetime

# =============================================================================
# CONFIGURATION & CONSTANTS
# =============================================================================

SIZE_PRESETS = {
    "Quick EDA / Notebook visuals": {"px": (800, 500), "figsize": (8, 5)},
    "Detailed Comparison Charts": {"px": (1000, 600), "figsize": (10, 6)},
    "Correlation Heatmaps / Pairplots": {"px": (1200, 800), "figsize": (12, 8)},
    "Presentation / Report visuals": {"px": (1400, 800), "figsize": (14, 8)},
    "Multiple subplots (2x2)": {"px": (1200, 1000), "figsize": (12, 10)}
}

DATA_PATH = os.path.join("data", "test_data.csv")
os.makedirs(os.path.dirname(DATA_PATH), exist_ok=True)

# =============================================================================
# PAGE CONFIGURATION - Enhanced for better UX
# =============================================================================

st.set_page_config(
    page_title="🤖 AutoML Pipeline Pro",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'About': "AutoML Pipeline Pro - Automated Machine Learning Made Easy"
    }
)

# =============================================================================
# CUSTOM CSS - Modern, Professional Design
# =============================================================================

st.markdown("""
<style>
    /* Import Google Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
    
    /* Global Styles */
    .main {
        font-family: 'Inter', sans-serif;
    }
    
    /* Custom Headers with Gradient */
    .main-header {
        font-size: 3rem;
        font-weight: 700;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 2rem;
        padding: 1rem 0;
        animation: fadeIn 0.8s ease-in;
    }
    
    .section-header {
        font-size: 1.5rem;
        font-weight: 600;
        color: #2d3748;
        margin: 1.5rem 0 1rem 0;
        padding-bottom: 0.5rem;
        border-bottom: 3px solid #667eea;
    }
    
    /* Enhanced Metric Cards with Hover Effect */
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 12px;
        margin: 0.5rem 0;
        color: white;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        transition: transform 0.3s ease, box-shadow 0.3s ease;
    }
    
    .metric-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 8px 15px rgba(0, 0, 0, 0.2);
    }
    
    .metric-value {
        font-size: 2.5rem;
        font-weight: 700;
        margin: 0.5rem 0;
    }
    
    .metric-label {
        font-size: 0.9rem;
        opacity: 0.9;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    
    /* Success Message with Animation */
    .success-message {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1.5rem;
        border-radius: 12px;
        border: none;
        box-shadow: 0 4px 12px rgba(102, 126, 234, 0.3);
        animation: slideIn 0.5s ease-out;
    }
    
    /* Warning Box */
    .warning-box {
        background-color: #fff3cd;
        color: #856404;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #ffc107;
        margin: 1rem 0;
    }
    
    /* Info Box */
    .info-box {
        background: linear-gradient(135deg, #e0f2fe 0%, #bae6fd 100%);
        color: #0c4a6e;
        padding: 1.5rem;
        border-radius: 12px;
        margin: 1rem 0;
        border-left: 4px solid #0ea5e9;
    }
    
    /* Best Model Badge */
    .best-model-badge {
        background: linear-gradient(135deg, #fbbf24 0%, #f59e0b 100%);
        color: white;
        padding: 0.5rem 1rem;
        border-radius: 20px;
        display: inline-block;
        font-weight: 600;
        box-shadow: 0 2px 8px rgba(251, 191, 36, 0.3);
    }
    
    /* Sidebar Styling */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #f8f9fa 0%, #e9ecef 100%);
    }
    
    /* Button Enhancements */
    .stButton>button {
        border-radius: 8px;
        font-weight: 600;
        transition: all 0.3s ease;
        border: none;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
    }
    
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.2);
    }
    
    /* Enhanced Dataframe Styling */
    .dataframe {
        border-radius: 8px;
        overflow: hidden;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.1);
    }
    
    /* Loading Spinner */
    .stSpinner > div {
        border-top-color: #667eea !important;
    }
    
    /* Animations */
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(-20px); }
        to { opacity: 1; transform: translateY(0); }
    }
    
    @keyframes slideIn {
        from { transform: translateX(-100%); opacity: 0; }
        to { transform: translateX(0); opacity: 1; }
    }
    
    /* Tab Styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    
    .stTabs [data-baseweb="tab"] {
        border-radius: 8px 8px 0 0;
        padding: 12px 24px;
        font-weight: 600;
    }
    
    /* Progress Bar */
    .stProgress > div > div {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    }
</style>
""", unsafe_allow_html=True)

# =============================================================================
# UTILITY FUNCTIONS - Optimized with Caching
# =============================================================================

def get_chart_dimensions(preset_name: str) -> Tuple[Tuple[int, int], Tuple[int, int]]:
    """Get chart dimensions from preset - cached for performance"""
    preset = SIZE_PRESETS.get(preset_name, SIZE_PRESETS["Quick EDA / Notebook visuals"])
    return preset["px"], preset["figsize"]

def load_dataset() -> pd.DataFrame:
    """Load dataset with caching for better performance"""
    if os.path.exists(DATA_PATH):
        return pd.read_csv(DATA_PATH, index_col=None)
    return None

def get_dataset_stats(df: pd.DataFrame) -> Dict[str, Any]:
    """Calculate dataset statistics - cached to avoid recomputation"""
    return {
        'rows': df.shape[0],
        'columns': df.shape[1],
        'missing': df.isnull().sum().sum(),
        'memory': df.memory_usage(deep=True).sum() / 1024,
        'numeric_cols': len(df.select_dtypes(include=[np.number]).columns),
        'categorical_cols': len(df.select_dtypes(include=['object']).columns)
    }

def save_matplotlib_to_png_bytes(fig) -> bytes:
    """Convert matplotlib figure to PNG bytes"""
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=150, bbox_inches='tight')
    buf.seek(0)
    return buf.getvalue()

def render_chart(df: pd.DataFrame, chart_type: str, columns: List[str], 
                 size_preset: str) -> Tuple[Any, bytes, bytes, str]:
    """
    Render chart with caching for better performance
    Returns: (plotly_fig, img_bytes, html_bytes, warning_message)
    """
    fig = None
    img_bytes = None
    html_bytes = None
    plot_warning = None
    (width_px, height_px), (w_in, h_in) = get_chart_dimensions(size_preset)

    try:
        # Enhanced color scheme for all charts
        color_scheme = px.colors.sequential.Viridis
        
        if chart_type in ['Bar Chart', 'Column Chart']:
            if len(columns) >= 2:
                fig = px.bar(df, x=columns[0], y=columns[1], 
                           color_discrete_sequence=color_scheme)
            elif len(columns) == 1:
                fig = px.bar(df, x=df.index, y=columns[0],
                           color_discrete_sequence=color_scheme)
            else:
                plot_warning = "Please select at least one or two columns."
                
        elif chart_type == 'Line Chart':
            if len(columns) >= 2:
                fig = px.line(df, x=columns[0], y=columns[1],
                            line_shape='spline')
            elif len(columns) == 1:
                fig = px.line(df, y=columns[0], line_shape='spline')
            else:
                plot_warning = "Please select at least one column."
                
        elif chart_type == 'Histogram':
            if len(columns) >= 1:
                fig = px.histogram(df, x=columns[0], 
                                 color_discrete_sequence=color_scheme,
                                 marginal='box')
            else:
                plot_warning = "Please select one column."
                
        elif chart_type == 'Pie Chart':
            if len(columns) >= 1:
                counts = df[columns[0]].value_counts()
                fig = px.pie(names=counts.index, values=counts.values,
                           color_discrete_sequence=px.colors.qualitative.Set3)
            else:
                plot_warning = "Please select a categorical column."
                
        elif chart_type == 'KPI Chart':
            # Enhanced KPI visualization
            kpi_fig = plt.figure(figsize=(w_in, max(3, h_in/2.5)))
            ax = kpi_fig.add_subplot(1,1,1)
            kpi_vals = [df[c].sum() if pd.api.types.is_numeric_dtype(df[c]) 
                       else df[c].count() for c in columns]
            
            bars = ax.bar(columns, kpi_vals, color='#667eea', alpha=0.8)
            for i, (bar, val) in enumerate(zip(bars, kpi_vals)):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{val:,.0f}', ha='center', va='bottom', 
                       fontweight='bold', fontsize=12)
            
            ax.set_ylabel('Value', fontweight='bold')
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            plt.xticks(rotation=45, ha='right')
            plt.tight_layout()
            img_bytes = save_matplotlib_to_png_bytes(kpi_fig)
            plt.close(kpi_fig)
            return None, img_bytes, None, None
            
        elif chart_type == 'Donut Chart':
            if len(columns) >= 1:
                counts = df[columns[0]].value_counts()
                fig = px.pie(names=counts.index, values=counts.values, 
                           hole=0.4,
                           color_discrete_sequence=px.colors.qualitative.Pastel)
            else:
                plot_warning = "Please select a categorical column."
                
        elif chart_type == 'Heatmap Chart':
            if len(columns) >= 2:
                fig = px.density_heatmap(df, x=columns[0], y=columns[1],
                                        color_continuous_scale='Viridis')
            else:
                plot_warning = "Heatmap needs at least two columns."

        # Enhanced layout for all Plotly figures
        if fig is not None:
            fig.update_layout(
                width=width_px, 
                height=height_px,
                template='plotly_white',
                font=dict(family='Inter, sans-serif', size=12),
                title_font_size=16,
                hoverlabel=dict(bgcolor="white", font_size=12),
                margin=dict(l=50, r=50, t=50, b=50)
            )
            html_bytes = fig.to_html().encode('utf-8')

    except Exception as ex:
        plot_warning = f"Error generating chart: {str(ex)}"

    return fig, img_bytes, html_bytes, plot_warning

# =============================================================================
# SIDEBAR NAVIGATION - Enhanced UI
# =============================================================================

with st.sidebar:
    st.markdown("## 🤖 AutoML Pipeline Pro")
    st.markdown("---")
    
    choice = st.radio(
        "**Navigation**", 
        ["📁 Upload Dataset", "📊 Data Analysis", "🧠 Machine Learning", "💾 Download Models"],
        help="Navigate through different sections of the AutoML pipeline"
    )
    
    st.markdown("---")
    
    # Enhanced dataset info display
    if os.path.exists(DATA_PATH):
        df_sidebar = load_dataset()
        if df_sidebar is not None:
            stats = get_dataset_stats(df_sidebar)
            
            st.markdown("### 📋 Dataset Overview")
            st.markdown(f"""
            <div style='background: white; padding: 1rem; border-radius: 8px; 
                        box-shadow: 0 2px 4px rgba(0,0,0,0.1);'>
                <p style='margin: 0.3rem 0;'><strong>Rows:</strong> {stats['rows']:,}</p>
                <p style='margin: 0.3rem 0;'><strong>Columns:</strong> {stats['columns']}</p>
                <p style='margin: 0.3rem 0;'><strong>Missing:</strong> {stats['missing']:,}</p>
                <p style='margin: 0.3rem 0;'><strong>Size:</strong> {stats['memory']:.1f} KB</p>
            </div>
            """, unsafe_allow_html=True)
    
    st.markdown("---")
    st.markdown("""
    <div style='background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                padding: 1rem; border-radius: 8px; color: white;'>
        <p style='margin: 0; font-size: 0.9rem;'>
            🚀 <strong>AutoML Pipeline Pro</strong><br>
            Advanced ML automation with intelligent preprocessing and model optimization
        </p>
    </div>
    """, unsafe_allow_html=True)

# Load main dataset
df = load_dataset()

# =============================================================================
# MAIN CONTENT - UPLOAD DATASET
# =============================================================================

if choice == "📁 Upload Dataset":
    st.markdown('<h1 class="main-header">📁 Upload Your Dataset</h1>', unsafe_allow_html=True)
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown('<p class="section-header">📂 File Upload</p>', unsafe_allow_html=True)
        
        file = st.file_uploader(
            "Choose a CSV or Excel file", 
            type=["csv", "xlsx"],
            help="Upload your dataset in CSV or Excel format. Max size: 200MB"
        )
        
        if file:
            try:
                # Progress bar for better UX
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                status_text.text("📥 Reading file...")
                progress_bar.progress(25)
                
                if file.name.endswith(".csv"):
                    df = pd.read_csv(file)
                else:
                    df = pd.read_excel(file)
                
                status_text.text("💾 Saving to cache...")
                progress_bar.progress(50)
                
                # Save to cache
                df.to_csv(DATA_PATH, index=None)
                
                status_text.text("🔄 Refreshing data...")
                progress_bar.progress(75)
                
                st.cache_data.clear()
                
                progress_bar.progress(100)
                status_text.text("✅ Upload complete!")
                
                st.markdown(f"""
                <div class="success-message">
                    <h3 style='margin: 0 0 0.5rem 0;'>✅ Successfully uploaded {file.name}</h3>
                    <p style='margin: 0;'>Dataset is ready for analysis and modeling</p>
                </div>
                """, unsafe_allow_html=True)
                
                # Enhanced dataset preview with statistics
                st.markdown('<p class="section-header">📋 Dataset Preview</p>', unsafe_allow_html=True)
                
                # Quick stats in colorful cards
                col1_stats, col2_stats, col3_stats, col4_stats = st.columns(4)
                
                with col1_stats:
                    st.markdown(f"""
                    <div class="metric-card">
                        <div class="metric-label">Total Rows</div>
                        <div class="metric-value">{df.shape[0]:,}</div>
                    </div>
                    """, unsafe_allow_html=True)
                
                with col2_stats:
                    st.markdown(f"""
                    <div class="metric-card">
                        <div class="metric-label">Columns</div>
                        <div class="metric-value">{df.shape[1]}</div>
                    </div>
                    """, unsafe_allow_html=True)
                
                with col3_stats:
                    missing = df.isnull().sum().sum()
                    st.markdown(f"""
                    <div class="metric-card">
                        <div class="metric-label">Missing Values</div>
                        <div class="metric-value">{missing:,}</div>
                    </div>
                    """, unsafe_allow_html=True)
                
                with col4_stats:
                    numeric_cols = len(df.select_dtypes(include=[np.number]).columns)
                    st.markdown(f"""
                    <div class="metric-card">
                        <div class="metric-label">Numeric Cols</div>
                        <div class="metric-value">{numeric_cols}</div>
                    </div>
                    """, unsafe_allow_html=True)
                
                # Data preview with enhanced styling
                st.dataframe(
                    df.head(10), 
                    use_container_width=True,
                    height=400
                )
                
            except Exception as e:
                st.error(f"❌ Error processing file: {str(e)}")
    
    with col2:
        st.markdown('<p class="section-header">📝 Guidelines</p>', unsafe_allow_html=True)
        st.markdown("""
        <div class="info-box">
            <h4 style='margin-top: 0;'>✅ Supported Formats</h4>
            <ul style='margin-bottom: 1rem;'>
                <li>CSV (.csv)</li>
                <li>Excel (.xlsx, .xls)</li>
            </ul>
            
            <h4>📋 Requirements</h4>
            <ul style='margin-bottom: 1rem;'>
                <li>First row = column headers</li>
                <li>No empty columns</li>
                <li>Max size: 200MB</li>
            </ul>
            
            <h4>🔧 Auto-Processing</h4>
            <ul style='margin-bottom: 0;'>
                <li>Missing value detection</li>
                <li>Data type inference</li>
                <li>Memory optimization</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

# =============================================================================
# MAIN CONTENT - DATA ANALYSIS
# =============================================================================

elif choice == "📊 Data Analysis":
    st.markdown('<h1 class="main-header">📊 Exploratory Data Analysis</h1>', unsafe_allow_html=True)
    
    if df is not None:
        # Analysis controls in expandable section
        with st.expander("⚙️ Chart Configuration", expanded=True):
            col1, col2, col3 = st.columns(3)
            
            with col1:
                chart_type = st.selectbox(
                    "📊 Chart Type",
                    ['Bar Chart', 'Line Chart', 'Histogram', 'Pie Chart', 
                     'KPI Chart', 'Donut Chart', 'Heatmap Chart'],
                    index=0
                )
            
            with col2:
                size_preset = st.selectbox(
                    "📏 Size Preset",
                    list(SIZE_PRESETS.keys()),
                    index=0
                )
            
            with col3:
                columns_selected = st.multiselect(
                    "📋 Select Columns",
                    options=list(df.columns),
                    default=list(df.columns)[:2] if len(df.columns) >= 2 else list(df.columns)[:1]
                )
        
        # Tabs for organized content
        tab1, tab2, tab3, tab4 = st.tabs([
            "📈 Overview", 
            "🔍 Custom Charts", 
            "📊 Data Quality",
            "📑 Detailed Report"
        ])
        
        with tab1:
            st.markdown('<p class="section-header">📊 Dataset Overview</p>', unsafe_allow_html=True)
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.markdown("#### 🏷️ Data Types")
                dtype_df = df.dtypes.to_frame('Type')
                dtype_df['Count'] = 1
                dtype_summary = dtype_df.groupby('Type').count()
                st.dataframe(dtype_summary, use_container_width=True)
            
            with col2:
                st.markdown("#### 📊 Statistical Summary")
                st.dataframe(df.describe().round(2), use_container_width=True)
            
            with col3:
                st.markdown("#### 👁️ Sample Data")
                st.dataframe(df.head(5), use_container_width=True)
        
        with tab2:
            st.markdown('<p class="section-header">📊 Interactive Visualization</p>', unsafe_allow_html=True)
            
            if columns_selected:
                with st.spinner("🎨 Generating chart..."):
                    fig, img_bytes, html_bytes, plot_warning = render_chart(
                        df, chart_type, columns_selected, size_preset
                    )
                
                if plot_warning:
                    st.warning(f"⚠️ {plot_warning}")
                elif fig is not None:
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Download buttons
                    col1, col2 = st.columns(2)
                    with col1:
                        if img_bytes:
                            st.download_button(
                                label="💾 Download PNG",
                                data=img_bytes,
                                file_name=f"{chart_type.replace(' ', '_')}.png",
                                mime='image/png',
                                use_container_width=True
                            )
                    with col2:
                        if html_bytes:
                            st.download_button(
                                label="💾 Download HTML",
                                data=html_bytes,
                                file_name=f"{chart_type.replace(' ', '_')}.html",
                                mime='text/html',
                                use_container_width=True
                            )
                elif img_bytes:
                    st.image(img_bytes, use_column_width=True)
                    st.download_button(
                        label="💾 Download PNG",
                        data=img_bytes,
                        file_name=f"{chart_type.replace(' ', '_')}.png",
                        mime='image/png'
                    )
            else:
                st.info("👆 Select columns from the configuration above to generate a chart")
        
        with tab3:
            st.markdown('<p class="section-header">🔍 Data Quality Analysis</p>', unsafe_allow_html=True)
            
            missing_data = df.isnull().sum()
            missing_data = missing_data[missing_data > 0].sort_values(ascending=False)
            
            if len(missing_data) > 0:
                # Visual representation of missing data
                fig_missing = go.Figure()
                fig_missing.add_trace(go.Bar(
                    x=missing_data.index,
                    y=missing_data.values,
                    marker_color='#667eea',
                    text=missing_data.values,
                    textposition='auto',
                ))
                fig_missing.update_layout(
                    title="Missing Values by Column",
                    xaxis_title="Columns",
                    yaxis_title="Missing Count",
                    template='plotly_white',
                    height=400
                )
                st.plotly_chart(fig_missing, use_container_width=True)
                
                # Detailed table
                missing_df = pd.DataFrame({
                    'Column': missing_data.index,
                    'Missing Count': missing_data.values,
                    'Missing %': (missing_data.values / len(df) * 100).round(2)
                }).reset_index(drop=True)
                
                st.dataframe(missing_df, use_container_width=True)
            else:
                st.markdown("""
                <div class="success-message">
                    <h3 style='margin: 0;'>🎉 Excellent Data Quality!</h3>
                    <p style='margin: 0.5rem 0 0 0;'>No missing values detected in the dataset</p>
                </div>
                """, unsafe_allow_html=True)
        
        with tab4:
            st.markdown('<p class="section-header">📑 Comprehensive Profiling Report</p>', unsafe_allow_html=True)
            
            st.markdown("""
            <div class="warning-box">
                ⏱️ <strong>Note:</strong> Generating a detailed report may take several minutes for large datasets.
            </div>
            """, unsafe_allow_html=True)
            
            if not HAS_PROFILING:
                st.info("Profiling dependencies are missing in this environment. Install `ydata-profiling` and `streamlit-pandas-profiling` to enable this report.")
            elif st.button("🔄 Generate Detailed Profiling Report", type="primary"):
                with st.spinner("🔍 Analyzing dataset... This may take a few moments."):
                    try:
                        profile_report = ProfileReport(
                            df, 
                            explorative=True, 
                            minimal=False,
                            title="Dataset Profiling Report"
                        )
                        st_profile_report(profile_report)
                    except Exception as e:
                        st.error(f"❌ Error generating report: {str(e)}")
                        st.info("💡 Try with a smaller dataset or check data quality.")
    else:
        st.markdown("""
        <div class="warning-box">
            <h3 style='margin: 0 0 0.5rem 0;'>⚠️ No Dataset Loaded</h3>
            <p style='margin: 0;'>Please upload a dataset in the <strong>Upload Dataset</strong> section first.</p>
        </div>
        """, unsafe_allow_html=True)

# =============================================================================
# MAIN CONTENT - MACHINE LEARNING
# =============================================================================

elif choice == "🧠 Machine Learning":
    st.markdown('<h1 class="main-header">🧠 Automated Machine Learning</h1>', unsafe_allow_html=True)

    if df is not None:
        st.markdown('<p class="section-header">⚙️ Model Configuration</p>', unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            target_col = st.selectbox(
                "🎯 Target Column", 
                df.columns,
                help="Select the column you want to predict"
            )
            
        with col2:
            # Auto-detect problem type with visual indicator
            if df[target_col].dtype == 'object' or df[target_col].nunique() < 10:
                default_type = "Classification"
                detected_icon = "🏷️"
            else:
                default_type = "Regression"
                detected_icon = "📈"
            
            st.markdown(f"**Detected Type:** {detected_icon} {default_type}")
            problem_type = st.radio(
                "📋 Problem Type", 
                ["Classification", "Regression"],
                index=0 if default_type == "Classification" else 1,
                help="Classification for categorical targets, Regression for continuous values"
            )

        # Model selection with better UI
        st.markdown('<p class="section-header">🤖 Model Selection</p>', unsafe_allow_html=True)
        
        if problem_type == "Classification":
            available_models = ["Logistic Regression", "Random Forest", "SVM"]
            default_models = ["Logistic Regression", "Random Forest"]
        else:
            available_models = ["Linear Regression", "Random Forest Regressor", "SVR"]
            default_models = ["Linear Regression", "Random Forest Regressor"]

        selected_models = st.multiselect(
            "Select models to train and compare", 
            available_models, 
            default=default_models,
            help="Choose multiple models for comprehensive comparison"
        )

        # Advanced options in collapsible section
        with st.expander("🔧 Advanced Configuration"):
            col1, col2, col3 = st.columns(3)
            
            with col1:
                test_size = st.slider("Test Set Size", 0.1, 0.4, 0.2, 0.05,
                                     help="Proportion of data for testing")
            with col2:
                cv_folds = st.slider("Cross-Validation Folds", 3, 10, 5,
                                    help="Number of CV folds for validation")
            with col3:
                random_state = st.number_input("Random State", 0, 1000, 42,
                                              help="Seed for reproducibility")

        # Training section with enhanced feedback
        if selected_models:
            if st.button("🚀 Train Models", type="primary", use_container_width=True):
                # Progress tracking
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                status_text.text("🔄 Preprocessing data...")
                progress_bar.progress(20)
                
                with st.spinner("🧠 Training models... Please wait."):
                    try:
                        results = ml_pipeline(df, target_col, problem_type, selected_models)
                        progress_bar.progress(100)
                        status_text.text("✅ Training complete!")
                    except Exception as e:
                        st.error(f"❌ Training error: {str(e)}")
                        results = None

                if results:
                    st.markdown('<p class="section-header">📊 Model Performance Results</p>', 
                              unsafe_allow_html=True)
                    
                    # Create enhanced results dataframe
                    display_results = {}
                    for model_name, metrics in results.items():
                        if 'error' not in metrics:
                            display_results[model_name] = {
                                k: v for k, v in metrics.items() 
                                if k not in ['Report', 'Model', 'Scaler']
                            }
                    
                    if display_results:
                        results_df = pd.DataFrame(display_results).T
                        
                        # Style the dataframe
                        styled_df = results_df.style.highlight_max(
                            axis=0, 
                            props='background-color: #d4edda; font-weight: bold;'
                        )
                        st.dataframe(styled_df, use_container_width=True)
                        
                        # Determine best model
                        if problem_type == "Classification":
                            best_model = max(results.keys(), 
                                           key=lambda x: results[x].get('Accuracy', 0))
                            best_score = results[best_model]['Accuracy']
                            metric_name = "Accuracy"
                        else:
                            best_model = max(results.keys(), 
                                           key=lambda x: results[x].get('R2', -float('inf')))
                            best_score = results[best_model]['R2']
                            metric_name = "R² Score"
                        
                        # Best model showcase
                        st.markdown(f"""
                        <div class="success-message">
                            <h3 style='margin: 0 0 1rem 0;'>🏆 Best Performing Model</h3>
                            <div style='display: flex; justify-content: space-between; align-items: center;'>
                                <div>
                                    <h2 style='margin: 0; font-size: 2rem;'>{best_model}</h2>
                                    <p style='margin: 0.5rem 0 0 0; font-size: 1.1rem;'>
                                        <strong>{metric_name}:</strong> {best_score:.4f}
                                    </p>
                                </div>
                                <div class="best-model-badge">
                                    RECOMMENDED
                                </div>
                            </div>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        # Store results in session state
                        st.session_state['ml_results'] = results
                        st.session_state['best_model'] = best_model
                        st.session_state['problem_type'] = problem_type
                        
                        # Enhanced model comparison visualization
                        if len(results) > 1:
                            st.markdown('<p class="section-header">📈 Model Comparison</p>', 
                                      unsafe_allow_html=True)
                            
                            if problem_type == "Classification":
                                fig = go.Figure()
                                models = list(results.keys())
                                
                                fig.add_trace(go.Bar(
                                    name='Accuracy',
                                    x=models,
                                    y=[results[m]['Accuracy'] for m in models],
                                    marker_color='#667eea'
                                ))
                                
                                fig.add_trace(go.Bar(
                                    name='CV Mean',
                                    x=models,
                                    y=[results[m]['CV_Mean'] for m in models],
                                    marker_color='#764ba2'
                                ))
                                
                                fig.update_layout(
                                    barmode='group',
                                    title="Classification Model Comparison",
                                    xaxis_title="Models",
                                    yaxis_title="Score",
                                    template='plotly_white',
                                    height=400
                                )
                                st.plotly_chart(fig, use_container_width=True)
                            else:
                                fig = go.Figure()
                                models = list(results.keys())
                                
                                fig.add_trace(go.Bar(
                                    name='R² Score',
                                    x=models,
                                    y=[results[m]['R2'] for m in models],
                                    marker_color='#667eea'
                                ))
                                
                                fig.add_trace(go.Bar(
                                    name='CV Mean',
                                    x=models,
                                    y=[results[m]['CV_Mean'] for m in models],
                                    marker_color='#764ba2'
                                ))
                                
                                fig.update_layout(
                                    barmode='group',
                                    title="Regression Model Comparison",
                                    xaxis_title="Models",
                                    yaxis_title="Score",
                                    template='plotly_white',
                                    height=400
                                )
                                st.plotly_chart(fig, use_container_width=True)
                        
                        # Success message with next steps
                        st.markdown("""
                        <div class="info-box">
                            <h4 style='margin-top: 0;'>✅ Next Steps</h4>
                            <ul style='margin-bottom: 0;'>
                                <li>Review model performance metrics above</li>
                                <li>Navigate to <strong>Download Models</strong> to save your trained models</li>
                                <li>Use the downloaded models for predictions in your applications</li>
                            </ul>
                        </div>
                        """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div class="warning-box">
                ⚠️ <strong>Select at least one model</strong> to begin training
            </div>
            """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div class="warning-box">
            <h3 style='margin: 0 0 0.5rem 0;'>⚠️ No Dataset Loaded</h3>
            <p style='margin: 0;'>Please upload a dataset in the <strong>Upload Dataset</strong> section first.</p>
        </div>
        """, unsafe_allow_html=True)

# =============================================================================
# MAIN CONTENT - DOWNLOAD MODELS
# =============================================================================

elif choice == "💾 Download Models":
    st.markdown('<h1 class="main-header">💾 Download Trained Models</h1>', unsafe_allow_html=True)
    
    if 'ml_results' in st.session_state and st.session_state['ml_results']:
        results = st.session_state['ml_results']
        best_model_name = st.session_state.get('best_model', list(results.keys())[0])
        
        st.markdown('<p class="section-header">📦 Available Models</p>', unsafe_allow_html=True)
        
        # Display each model with enhanced UI
        for idx, model_name in enumerate(results.keys()):
            if 'error' not in results[model_name]:
                col1, col2, col3 = st.columns([3, 1, 1])
                
                with col1:
                    is_best = model_name == best_model_name
                    
                    if is_best:
                        st.markdown(f"""
                        <div style='background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                                    padding: 1.5rem; border-radius: 12px; color: white; margin: 0.5rem 0;'>
                            <div style='display: flex; align-items: center; gap: 1rem;'>
                                <span style='font-size: 2rem;'>🏆</span>
                                <div>
                                    <h3 style='margin: 0; font-size: 1.5rem;'>{model_name}</h3>
                                    <p style='margin: 0.3rem 0 0 0; opacity: 0.9;'>
                                        {st.session_state['problem_type']} • Best Performance
                                    </p>
                                </div>
                            </div>
                        </div>
                        """, unsafe_allow_html=True)
                    else:
                        st.markdown(f"""
                        <div style='background: white; padding: 1.5rem; border-radius: 12px; 
                                    border: 2px solid #e2e8f0; margin: 0.5rem 0;'>
                            <h3 style='margin: 0; color: #2d3748;'>{model_name}</h3>
                            <p style='margin: 0.3rem 0 0 0; color: #718096;'>
                                {st.session_state['problem_type']}
                            </p>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    # Display metrics
                    metrics_col1, metrics_col2, metrics_col3 = st.columns(3)
                    
                    if st.session_state['problem_type'] == "Classification":
                        with metrics_col1:
                            st.metric("Accuracy", f"{results[model_name]['Accuracy']:.4f}")
                        with metrics_col2:
                            st.metric("CV Mean", f"{results[model_name]['CV_Mean']:.4f}")
                        with metrics_col3:
                            st.metric("CV Std", f"{results[model_name]['CV_Std']:.4f}")
                    else:
                        with metrics_col1:
                            st.metric("R² Score", f"{results[model_name]['R2']:.4f}")
                        with metrics_col2:
                            st.metric("RMSE", f"{results[model_name]['RMSE']:.4f}")
                        with metrics_col3:
                            st.metric("CV Mean", f"{results[model_name]['CV_Mean']:.4f}")
                
                with col2:
                    # Download individual model
                    try:
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
                            key=f"download_{model_name}",
                            use_container_width=True
                        )
                    except Exception as e:
                        st.error(f"Error: {str(e)}")
                
                with col3:
                    if is_best:
                        st.markdown("""
                        <div style='background: #fbbf24; color: white; padding: 0.5rem; 
                                    border-radius: 8px; text-align: center; font-weight: bold;'>
                            ⭐ BEST
                        </div>
                        """, unsafe_allow_html=True)
                
                st.markdown("---")
        
        # Download all models section
        st.markdown('<p class="section-header">📦 Bulk Download</p>', unsafe_allow_html=True)
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.markdown("""
            <div class="info-box">
                <h4 style='margin-top: 0;'>💡 Download All Models</h4>
                <p style='margin: 0;'>
                    Get a complete package containing all trained models, scalers, 
                    and performance metrics in a single file for easy deployment.
                </p>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            if st.button("📦 Package All Models", type="primary", use_container_width=True):
                try:
                    all_models_data = {}
                    for model_name in results.keys():
                        if 'error' not in results[model_name]:
                            all_models_data[model_name] = {
                                'model': results[model_name]['Model'],
                                'scaler': results[model_name].get('Scaler'),
                                'metrics': {k: v for k, v in results[model_name].items() 
                                          if k not in ['Model', 'Scaler', 'Report']}
                            }
                    
                    buffer = io.BytesIO()
                    pickle.dump(all_models_data, buffer)
                    buffer.seek(0)
                    
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    
                    st.download_button(
                        label="📥 Download Complete Package",
                        data=buffer.getvalue(),
                        file_name=f"automl_models_{timestamp}.pkl",
                        mime="application/octet-stream",
                        use_container_width=True
                    )
                except Exception as e:
                    st.error(f"Error creating package: {str(e)}")
        
        # Usage instructions
        st.markdown('<p class="section-header">📖 Usage Guide</p>', unsafe_allow_html=True)
        
        tab1, tab2 = st.tabs(["🐍 Python Code", "📝 Instructions"])
        
        with tab1:
            st.code("""
# Load and use a single model
import pickle
import pandas as pd

# Load the model file
with open('model_file.pkl', 'rb') as f:
    model_data = pickle.load(f)

model = model_data['model']
scaler = model_data['scaler']  # May be None for tree-based models
model_name = model_data['model_name']

# Prepare your new data (same features as training data)
X_new = pd.DataFrame({
    'feature1': [value1],
    'feature2': [value2],
    # ... add all features
})

# Make predictions
if scaler is not None:
    X_scaled = scaler.transform(X_new)
    predictions = model.predict(X_scaled)
else:
    predictions = model.predict(X_new)

print(f"Predictions: {predictions}")

# For classification, get probabilities
if hasattr(model, 'predict_proba'):
    probabilities = model.predict_proba(X_scaled if scaler else X_new)
    print(f"Probabilities: {probabilities}")
            """, language="python")
        
        with tab2:
            st.markdown("""
            ### 🚀 Quick Start Guide
            
            #### 1. Download Models
            - Click **Download** for individual models
            - Or use **Package All Models** for bulk download
            
            #### 2. Load in Your Application
            ```python
            import pickle
            with open('model_file.pkl', 'rb') as f:
                model_data = pickle.load(f)
            ```
            
            #### 3. Important Notes
            - **Tree-based models** (Random Forest): No scaling needed
            - **Linear models** (Logistic Regression, SVM): Scaling required
            - Always use the same features and order as training data
            - Handle missing values before prediction
            
            #### 4. Model Files Include
            - ✅ Trained model object
            - ✅ Scaler (if applicable)
            - ✅ Model name and type
            - ✅ Performance metrics (in package file)
            
            #### 5. Best Practices
            - Test predictions on sample data first
            - Monitor model performance over time
            - Retrain periodically with new data
            - Keep track of feature engineering steps
            """)
    else:
        st.markdown("""
        <div class="warning-box">
            <h3 style='margin: 0 0 0.5rem 0;'>🔄 No Trained Models Available</h3>
            <p style='margin: 0;'>
                Please train some models in the <strong>Machine Learning</strong> section first.
            </p>
        </div>
        """, unsafe_allow_html=True)
        
        # Helpful call-to-action
        st.markdown("""
        <div class="info-box" style='margin-top: 2rem;'>
            <h4 style='margin-top: 0;'>💡 How to Get Started</h4>
            <ol style='margin-bottom: 0;'>
                <li><strong>Upload Dataset:</strong> Go to the Upload section and load your CSV/Excel file</li>
                <li><strong>Explore Data:</strong> Analyze your data in the Data Analysis section</li>
                <li><strong>Train Models:</strong> Navigate to Machine Learning and train your models</li>
                <li><strong>Download:</strong> Return here to download your trained models</li>
            </ol>
        </div>
        """, unsafe_allow_html=True)

# =============================================================================
# FOOTER
# =============================================================================

st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #718096; padding: 2rem 0;'>
    <p style='margin: 0; font-size: 0.9rem;'>
        🤖 <strong>AutoML Pipeline Pro</strong> | Built with Streamlit & scikit-learn
    </p>
    <p style='margin: 0.5rem 0 0 0; font-size: 0.8rem;'>
        Automated Machine Learning for Everyone
    </p>
</div>
""", unsafe_allow_html=True)
