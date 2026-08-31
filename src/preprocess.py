"""
Shared preprocessing utilities for anomaly detection.

Unsupervised anomaly detection is sensitive to scaling and nulls -- an
unscaled column with large values will dominate distance/reconstruction
calculations, so numeric columns are always scaled before fitting
Isolation Forest or the Autoencoder.

IMPORTANT: one-hot encoded dummy columns are intentionally left
unscaled (raw 0/1). Applying StandardScaler to a binary column that's
mostly 0s with a rare 1 gives that rare category a tiny standard
deviation, so its scaled value can blow up to 10, 50, even 180+ --
completely unrelated to whether the row is a genuine anomaly. That
inflated value then dominates SHAP/reconstruction-error attribution,
making "belongs to a rare category" look identical to "is a real
anomaly." Scaling is only meaningful for continuous magnitudes, not
presence/absence indicators, so dummy columns are excluded from it.
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer


def preprocess(df: pd.DataFrame, numeric_cols, categorical_cols=None):
    """
    Clean, encode, and scale a DataFrame for anomaly detection.

    Args:
        df: raw input DataFrame
        numeric_cols: list of numeric column names to use
        categorical_cols: optional list of categorical column names to
            one-hot encode

    Returns:
        X: numpy array -- numeric columns are standard-scaled, one-hot
            dummy columns are left as raw 0/1
        scaler: fitted StandardScaler (fit only on the numeric columns --
            needed to transform new numeric data later)
        feature_names: list of column names matching X's columns, in order
            (numeric columns first, then dummy columns)
    """
    data = df.copy()

    # Replace common null markers seen across these datasets
    data.replace(['?', 'None', 'none', 'NA', ''], np.nan, inplace=True)

    # Impute numeric nulls with median
    imputer = SimpleImputer(strategy='median')
    data[numeric_cols] = imputer.fit_transform(data[numeric_cols])

    if categorical_cols:
        data = pd.get_dummies(data, columns=categorical_cols, drop_first=True)
        dummy_cols = [c for c in data.columns if c not in numeric_cols and c not in df.columns]
    else:
        dummy_cols = []

    all_cols = numeric_cols + dummy_cols

    # Scale only the numeric columns
    scaler = StandardScaler()
    X_numeric = scaler.fit_transform(data[numeric_cols])

    if dummy_cols:
        # Leave dummy columns as raw 0/1 -- no scaling
        X_dummy = data[dummy_cols].to_numpy(dtype=float)
        X = np.hstack([X_numeric, X_dummy])
    else:
        X = X_numeric

    return X, scaler, all_cols