"""

For Isolation Forest, uses SHAP's TreeExplainer -- computed only on the
handful of rows being explained (not the full dataset) since SHAP over
hundreds of thousands of rows is impractically slow.

For the Autoencoder, uses per-feature squared reconstruction error
(src/explain.py's explain_autoencoder) -- this is NOT SHAP, and is
described accurately as "feature-level reconstruction error attribution."

Run from the project root:

    python scripts/explain_anomalies.py

Requires models/<domain>_isolation.joblib and models/<domain>_autoencoder.pt
to already exist.
"""

import sys
from pathlib import Path

# Allow running via `python scripts/explain_anomalies.py` from the project root
sys.path.append(str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
import shap
import torch

from src.autoencoder import Autoencoder
from src.config import CONFIGS
from src.explain import explain_autoencoder
from src.isolation import IsolationDetector
from src.preprocess import preprocess

FILES = {
    'healthcare': 'data/healthcare/diabetic_data.csv',
    'financial': 'data/financial/creditcard.csv',
    'realestate': 'data/realestate/train.csv',
}

MODELS_DIR = 'models'
TOP_N = 5           # how many top anomalies to explain per method
TOP_FEATURES = 5    # how many contributing features to show per anomaly


def raw_value(df, row_idx, col_name, scaled_val):
    """Show the original (human-readable) value when possible; fall back to
    the scaled value for one-hot dummy columns that don't exist in the
    original DataFrame."""
    if col_name in df.columns:
        return str(df.iloc[row_idx][col_name])
    return f'(encoded category, scaled value={scaled_val:.2f})'


def explain_isolation_top_n(domain, df, X, cols):
    model_path = f'{MODELS_DIR}/{domain}_isolation.joblib'
    detector = IsolationDetector().load(model_path)
    scores = detector.score(X)

    top_idx = np.argsort(scores)[:TOP_N]  # most negative score = most anomalous
    X_subset = X[top_idx]

    explainer = shap.TreeExplainer(detector.model)
    shap_vals = explainer.shap_values(X_subset)

    print(f'\n--- Isolation Forest: top {TOP_N} anomalies ---')
    for i, orig_idx in enumerate(top_idx):
        row_shap = shap_vals[i]
        top_feat_idx = np.argsort(np.abs(row_shap))[-TOP_FEATURES:][::-1]
        print(f'\nRow {orig_idx}  (anomaly score: {scores[orig_idx]:.4f}, more negative = more anomalous)')
        for fi in top_feat_idx:
            val = raw_value(df, orig_idx, cols[fi], X_subset[i, fi])
            direction = 'pushes toward anomaly' if row_shap[fi] < 0 else 'pushes toward normal'
            print(f'  {cols[fi]:30s} shap={row_shap[fi]:+.4f}  ({direction})  value={val}')


def explain_autoencoder_top_n(domain, df, X, cols):
    model_path = f'{MODELS_DIR}/{domain}_autoencoder.pt'
    model = Autoencoder(input_dim=X.shape[1])
    model.load_state_dict(torch.load(model_path))
    model.eval()

    X_tensor = torch.FloatTensor(X)
    errors = model.reconstruction_error(X_tensor)
    top_idx = np.argsort(errors)[::-1][:TOP_N]  # highest error = most anomalous

    print(f'\n--- Autoencoder: top {TOP_N} anomalies ---')
    for orig_idx in top_idx:
        top_feats = explain_autoencoder(model, X[orig_idx], cols)  # returns top 8 by default
        print(f'\nRow {orig_idx}  (reconstruction error: {errors[orig_idx]:.4f})')
        for name, err in top_feats[:TOP_FEATURES]:
            fi = cols.index(name)
            val = raw_value(df, orig_idx, name, X[orig_idx, fi])
            print(f'  {name:30s} recon_error={err:.4f}  value={val}')


def main():
    for domain, path in FILES.items():
        print(f'\n{"=" * 60}\n{domain.upper()}\n{"=" * 60}')
        df = pd.read_csv(path)
        cfg = CONFIGS[domain]
        X, scaler, cols = preprocess(df, cfg['numeric'], cfg.get('categorical', []))

        explain_isolation_top_n(domain, df, X, cols)
        explain_autoencoder_top_n(domain, df, X, cols)


if __name__ == '__main__':
    main()