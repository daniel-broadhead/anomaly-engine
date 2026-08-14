"""
Explainability layer.

- Isolation Forest: SHAP TreeExplainer (native scikit-learn tree support).
- Autoencoder: no native SHAP explainer, so we use per-feature squared
  reconstruction error as an interpretable, honest attribution method.
  This is NOT SHAP -- describe it accurately as "feature-level
  reconstruction error attribution" in your README and interviews.
"""

import joblib
import numpy as np
import shap
import torch


def explain_isolation(model_path, X, feature_names, sample_idx):
    model = joblib.load(model_path)
    explainer = shap.TreeExplainer(model)
    shap_vals = explainer.shap_values(X)

    row_shap = shap_vals[sample_idx]
    top_idx = np.argsort(np.abs(row_shap))[-8:][::-1]
    return [(feature_names[i], float(row_shap[i])) for i in top_idx]


def explain_autoencoder(model, X_row, feature_names):
    """
    Attribution by per-feature squared reconstruction error.
    Returns the top 8 contributing features for one anomalous row.
    """
    x_tensor = torch.FloatTensor(X_row).unsqueeze(0)
    with torch.no_grad():
        recon = model(x_tensor).squeeze(0).numpy()

    per_feature_error = (X_row - recon) ** 2
    top_idx = np.argsort(per_feature_error)[-8:][::-1]
    return [(feature_names[i], float(per_feature_error[i])) for i in top_idx]
