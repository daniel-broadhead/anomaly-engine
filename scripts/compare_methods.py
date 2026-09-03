"""
Compare Isolation Forest vs. Autoencoder anomaly flags across all domains.

Loads both trained models per domain, recomputes which rows each method
flags, and reports how much they agree. For financial, additionally
breaks the overlap down by true fraud status so you can see whether
combining both methods would catch more fraud than either alone.

Run from the project root:

    python scripts/compare_methods.py

Requires models/<domain>_isolation.joblib and models/<domain>_autoencoder.pt
to already exist.
"""

import sys
from pathlib import Path

# Allow running via `python scripts/compare_methods.py` from the project root
sys.path.append(str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
import torch

from src.autoencoder import Autoencoder
from src.config import CONFIGS, CONTAMINATION
from src.isolation import IsolationDetector
from src.preprocess import preprocess

FILES = {
    'healthcare': 'data/healthcare/diabetic_data.csv',
    'financial': 'data/financial/creditcard.csv',
    'realestate': 'data/realestate/train.csv',
}

MODELS_DIR = 'models'


def get_isolation_flags(domain, X):
    model_path = f'{MODELS_DIR}/{domain}_isolation.joblib'
    detector = IsolationDetector().load(model_path)
    preds = detector.predict(X)
    return preds == -1


def get_autoencoder_flags(domain, X):
    model_path = f'{MODELS_DIR}/{domain}_autoencoder.pt'
    model = Autoencoder(input_dim=X.shape[1])
    model.load_state_dict(torch.load(model_path))
    model.eval()

    X_tensor = torch.FloatTensor(X)
    errors = model.reconstruction_error(X_tensor)

    contamination = CONTAMINATION[domain]
    threshold = np.percentile(errors, 100 * (1 - contamination))
    return errors > threshold


def compare_domain(domain, path):
    print(f'\n=== {domain.upper()} ===')
    df = pd.read_csv(path)
    cfg = CONFIGS[domain]
    X, scaler, cols = preprocess(df, cfg['numeric'], cfg.get('categorical', []))

    iso_flagged = get_isolation_flags(domain, X)
    ae_flagged = get_autoencoder_flags(domain, X)

    both = iso_flagged & ae_flagged
    only_iso = iso_flagged & ~ae_flagged
    only_ae = ae_flagged & ~iso_flagged
    either = iso_flagged | ae_flagged

    n_iso = iso_flagged.sum()
    n_ae = ae_flagged.sum()
    n_both = both.sum()
    n_union = either.sum()
    jaccard = n_both / n_union if n_union > 0 else 0

    print(f'Isolation Forest flagged: {n_iso}')
    print(f'Autoencoder flagged:      {n_ae}')
    print(f'Flagged by both:          {n_both}')
    print(f'Flagged by either (union):{n_union}')
    print(f'Jaccard similarity:       {jaccard:.1%}')
    print(f'Of Isolation\'s flags, {n_both / n_iso:.1%} were also flagged by the Autoencoder')
    print(f'Of Autoencoder\'s flags, {n_both / n_ae:.1%} were also flagged by Isolation Forest')

    if domain == 'financial' and 'Class' in df.columns:
        y = df['Class'].values
        total_fraud = y.sum()

        fraud_both = (both & (y == 1)).sum()
        fraud_only_iso = (only_iso & (y == 1)).sum()
        fraud_only_ae = (only_ae & (y == 1)).sum()
        fraud_either = (either & (y == 1)).sum()
        fraud_missed = total_fraud - fraud_either

        print(f'\nFraud capture breakdown (total fraud cases: {total_fraud}):')
        print(f'  Caught by both methods:        {fraud_both}')
        print(f'  Caught only by Isolation Forest:{fraud_only_iso}')
        print(f'  Caught only by Autoencoder:     {fraud_only_ae}')
        print(f'  Caught by at least one (union): {fraud_either}  ({fraud_either / total_fraud:.1%} of all fraud)')
        print(f'  Missed by both:                 {fraud_missed}')

        union_precision = fraud_either / n_union if n_union > 0 else 0
        print(f'\n  If you flagged anything either method caught (union), precision would be {union_precision:.1%}')
        print(f'  and recall would be {fraud_either / total_fraud:.1%} -- vs. Isolation Forest alone at its own precision/recall.')

    return {
        'domain': domain,
        'n_iso': int(n_iso),
        'n_ae': int(n_ae),
        'n_both': int(n_both),
        'jaccard': jaccard,
    }


def main():
    results = []
    for domain, path in FILES.items():
        results.append(compare_domain(domain, path))

    print('\n' + '=' * 50)
    print('SUMMARY')
    print('=' * 50)
    for r in results:
        print(f"  {r['domain']:12s} iso={r['n_iso']:<6} ae={r['n_ae']:<6} both={r['n_both']:<6} jaccard={r['jaccard']:.1%}")


if __name__ == '__main__':
    main()