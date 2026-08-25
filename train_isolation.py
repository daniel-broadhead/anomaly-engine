"""
Phase 3: Train and validate Isolation Forest across all three domains.

Run from the project root:

    python train_isolation.py

For each domain this will:
  1. Load the CSV and run it through preprocess()
  2. Fit an IsolationDetector using the contamination rate from src/config.py
  3. Print how many anomalies were flagged
  4. Show the top 20 most anomalous rows for manual inspection
  5. Save the trained model to models/<domain>_isolation.joblib

For the financial domain specifically, it also checks flagged anomalies
against the ground-truth 'Class' column and reports precision (target: >30%,
per the guide).
"""

import numpy as np
import pandas as pd

from src.config import CONFIGS, CONTAMINATION
from src.isolation import IsolationDetector
from src.preprocess import preprocess

FILES = {
    'healthcare': 'data/healthcare/diabetic_data.csv',
    'financial': 'data/financial/creditcard.csv',
    'realestate': 'data/realestate/train.csv',
}

MODELS_DIR = 'models'


def train_domain(domain, path):
    print(f'\n=== {domain.upper()} ===')
    df = pd.read_csv(path)
    cfg = CONFIGS[domain]

    X, scaler, cols = preprocess(df, cfg['numeric'], cfg.get('categorical', []))
    contamination = CONTAMINATION[domain]

    detector = IsolationDetector(contamination=contamination)
    detector.fit(X)

    scores = detector.score(X)
    preds = detector.predict(X)
    n_anomalies = (preds == -1).sum()

    print(f'Contamination: {contamination}')
    print(f'Anomalies found: {n_anomalies} / {len(X)} ({n_anomalies / len(X):.2%})')

    # Show top 20 most anomalous rows (lowest score = most anomalous)
    top_idx = np.argsort(scores)[:20]
    display_cols = cfg['numeric'][:5]
    print(f'\nTop 20 anomalies (showing {display_cols}):')
    print(df.iloc[top_idx][display_cols].to_string())

    # Domain-specific validation
    if domain == 'financial' and 'Class' in df.columns:
        flagged = df.iloc[np.where(preds == -1)[0]]
        precision = flagged['Class'].mean()
        print(f'\nPrecision on fraud (flagged rows that are true fraud): {precision:.1%}')
        if precision < 0.30:
            print('  Below the 30% target -- consider tuning contamination or n_estimators.')
    else:
        print(f'\nNo ground-truth labels for {domain}. Manually review the rows above:')
        print('  does each flagged row make real-world sense as an outlier?')

    model_path = f'{MODELS_DIR}/{domain}_isolation.joblib'
    detector.save(model_path)
    print(f'\nSaved model: {model_path}')

    return {
        'domain': domain,
        'n_anomalies': int(n_anomalies),
        'contamination': contamination,
    }


def main():
    import os
    os.makedirs(MODELS_DIR, exist_ok=True)

    results = []
    for domain, path in FILES.items():
        results.append(train_domain(domain, path))

    print('\n' + '=' * 40)
    print('SUMMARY')
    print('=' * 40)
    for r in results:
        print(f"  {r['domain']:12s} contamination={r['contamination']:<6} anomalies={r['n_anomalies']}")


if __name__ == '__main__':
    main()