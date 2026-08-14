"""
Isolation Forest wrapper.

Fast, interpretable baseline for anomaly detection. No labels required --
it fits on the data distribution directly.
"""

import joblib
from sklearn.ensemble import IsolationForest


class IsolationDetector:
    def __init__(self, contamination=0.05, n_estimators=200, random_state=42):
        """
        contamination: estimated fraction of anomalies in the data
            (e.g. 0.05 = 5%). Tune per domain -- see src/config.py.
        """
        self.model = IsolationForest(
            contamination=contamination,
            n_estimators=n_estimators,
            random_state=random_state,
            n_jobs=-1
        )

    def fit(self, X):
        self.model.fit(X)
        return self

    def score(self, X):
        """Anomaly scores: more negative = more anomalous."""
        return self.model.decision_function(X)

    def predict(self, X):
        """-1 for anomaly, 1 for normal."""
        return self.model.predict(X)

    def save(self, path):
        joblib.dump(self.model, path)

    def load(self, path):
        self.model = joblib.load(path)
        return self
