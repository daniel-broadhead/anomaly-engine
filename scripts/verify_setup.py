"""
Phase 1/2 verification script.

Confirms all three datasets load correctly and pass through preprocess()
without errors. Run this from the project root:

    python scripts/verify_setup.py

If your downloaded filenames differ from the defaults below (Kaggle zips
sometimes rename things), edit the FILES dict to match what's actually in
your data/ folders.
"""

import sys
from pathlib import Path

# Allow running via `python scripts/verify_setup.py` from the project root
sys.path.append(str(Path(__file__).resolve().parent.parent))

import pandas as pd
from src.config import CONFIGS
from src.preprocess import preprocess

FILES = {
    'healthcare': 'data/healthcare/diabetic_data.csv',
    'financial': 'data/financial/creditcard.csv',
    'realestate': 'data/realestate/train.csv',
}


def main():
    all_passed = True

    for domain, path in FILES.items():
        print(f'\n=== {domain.upper()} ===')
        try:
            df = pd.read_csv(path)
        except FileNotFoundError:
            print(f'  FAILED to load: {path} not found. Check the filename/path.')
            all_passed = False
            continue

        print(f'  Loaded: {df.shape[0]} rows, {df.shape[1]} columns')
        print(f'  Dtypes: {dict(df.dtypes.value_counts())}')

        null_counts = df.isnull().sum().sort_values(ascending=False)
        top_nulls = null_counts[null_counts > 0].head(5)
        if len(top_nulls) > 0:
            print(f'  Top null columns:\n{top_nulls.to_string()}')
        else:
            print('  No nulls detected (may still have "?" or similar placeholder nulls).')

        cfg = CONFIGS[domain]
        missing_numeric = [c for c in cfg['numeric'] if c not in df.columns]
        missing_categorical = [c for c in cfg.get('categorical', []) if c not in df.columns]
        if missing_numeric or missing_categorical:
            print(f'  FAILED: config columns not found in this CSV.')
            if missing_numeric:
                print(f'    Missing numeric: {missing_numeric}')
            if missing_categorical:
                print(f'    Missing categorical: {missing_categorical}')
            print(f'  Actual columns available: {list(df.columns)}')
            all_passed = False
            continue

        try:
            X, scaler, cols = preprocess(df, cfg['numeric'], cfg.get('categorical', []))
            print(f'  preprocess() OK: output shape {X.shape}, {len(cols)} feature columns')
        except Exception as e:
            print(f'  FAILED in preprocess(): {e}')
            all_passed = False

    print('\n' + '=' * 40)
    print('ALL CHECKS PASSED' if all_passed else 'SOME CHECKS FAILED -- see details above')
    print('=' * 40)


if __name__ == '__main__':
    main()