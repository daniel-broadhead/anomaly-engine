"""
Shared preprocessing utilities for anomaly detection.

Unsupervised anomaly detection is sensitive to scaling and nulls -- an
unscaled column with large values will dominate distance/reconstruction
calculations, so always scale before fitting Isolation Forest or the
Autoencoder.
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
        X: scaled numpy array
        scaler: fitted StandardScaler (needed to transform new data later)
        feature_names: list of column names matching X's columns, in order
    """
    data = df.copy()

    # Replace common null markers seen across these datasets
    data.replace(['?', 'None', 'none', 'NA', ''], np.nan, inplace=True)

    # Impute numeric nulls with median
    imputer = SimpleImputer(strategy='median')
    data[numeric_cols] = imputer.fit_transform(data[numeric_cols])

    if categorical_cols:
        data = pd.get_dummies(data, columns=categorical_cols, drop_first=True)
        all_cols = numeric_cols + [c for c in data.columns if c not in numeric_cols and c not in df.columns]
    else:
        all_cols = numeric_cols

    scaler = StandardScaler()
    X = scaler.fit_transform(data[all_cols])

    return X, scaler, all_cols
