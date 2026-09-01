"""
Anomaly Detection Engine -- Streamlit demo app.

Upload a CSV, pick a domain and method, and get back a ranked list of
outlier rows with a score-distribution chart.

Run with: streamlit run app/streamlit_app.py
(from the project root, with models already trained -- see Phase 3/4)
"""

import sys
from pathlib import Path

# Allow running via `streamlit run app/streamlit_app.py` from repo root
sys.path.append(str(Path(__file__).resolve().parent.parent))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
import torch

from src.autoencoder import Autoencoder, score_with_autoencoder
from src.config import CONFIGS
from src.isolation import IsolationDetector
from src.preprocess import preprocess

MODELS_DIR = Path(__file__).resolve().parent.parent / 'models'

st.set_page_config(page_title='Anomaly Detection Engine', layout='wide')
st.title('Anomaly Detection Engine')
st.markdown('Upload a dataset, select a domain, and flag outliers with explanations.')

domain = st.sidebar.selectbox('Domain', list(CONFIGS.keys()))
method = st.sidebar.radio('Method', ['Isolation Forest', 'Autoencoder'])
top_n = st.sidebar.slider('Show top N anomalies', 5, 50, 20)

uploaded = st.file_uploader('Upload CSV', type='csv')

if uploaded:
    df = pd.read_csv(uploaded)
    cfg = CONFIGS[domain]

    missing_numeric = [c for c in cfg['numeric'] if c not in df.columns]
    missing_categorical = [c for c in cfg.get('categorical', []) if c not in df.columns]
    if missing_numeric or missing_categorical:
        st.error(
            f"This CSV doesn't match the **{domain}** domain's expected columns.\n\n"
            f"- Missing numeric columns: {missing_numeric or 'none'}\n"
            f"- Missing categorical columns: {missing_categorical or 'none'}\n\n"
            f"Double-check you've selected the right domain in the sidebar, or see `src/config.py` "
            f"for the expected schema."
        )
        st.stop()

    with st.spinner('Preprocessing...'):
        X, scaler, cols = preprocess(df, cfg['numeric'], cfg.get('categorical', []))

    if method == 'Isolation Forest':
        model_path = MODELS_DIR / f'{domain}_isolation.joblib'
        if not model_path.exists():
            st.error(f'No trained model found at {model_path}. Run Phase 3 training first.')
            st.stop()
        detector = IsolationDetector().load(str(model_path))
        scores = detector.score(X)
        preds = detector.predict(X)
    else:
        model_path = MODELS_DIR / f'{domain}_autoencoder.pt'
        if not model_path.exists():
            st.error(f'No trained model found at {model_path}. Run Phase 4 training first.')
            st.stop()
        model = Autoencoder(input_dim=X.shape[1])
        model.load_state_dict(torch.load(str(model_path)))
        model.eval()
        errors, preds = score_with_autoencoder(model, X)
        scores = -errors  # negate so higher = more anomalous, matching Isolation Forest sort order

    anomaly_idx = np.argsort(scores)[:top_n]

    st.subheader(f'Top {top_n} anomalies')
    st.dataframe(df.iloc[anomaly_idx])

    st.subheader('Anomaly score distribution')
    fig, ax = plt.subplots(figsize=(10, 3))
    ax.hist(scores, bins=50, color='steelblue', alpha=0.7)
    threshold = np.sort(scores)[top_n]
    ax.axvline(threshold, color='red', linestyle='--', label=f'Top {top_n} threshold')
    ax.set_xlabel('Anomaly score (lower = more anomalous)')
    ax.set_ylabel('Count')
    ax.legend()
    st.pyplot(fig)
else:
    st.info('Upload a CSV to get started. Column names must match the selected domain\'s config in src/config.py.')