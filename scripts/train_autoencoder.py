"""
Phase 4: Train and validate the PyTorch Autoencoder across all three domains.

Run from the project root:

    python scripts/train_autoencoder.py

For each domain this will:
  1. Load the CSV and run it through preprocess()
  2. Train an Autoencoder, printing loss every 10 epochs (confirm it's
     decreasing -- if it's flat, see the "Autoencoder loss does not
     decrease" troubleshooting entry in the guide)
  3. Score every row by reconstruction error and flag the top fraction
     matching that domain's contamination rate (from src/config.py)
  4. For financial, check flagged rows against the ground-truth 'Class'
     column and report precision
  5. For healthcare/real estate, print the top 20 flagged rows for manual
     domain-knowledge review
  6. Save the trained weights to models/<domain>_autoencoder.pt
"""

import sys
from pathlib import Path

# Allow running via `python scripts/train_autoencoder.py` from the project root
sys.path.append(str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from src.autoencoder import Autoencoder
from src.config import CONFIGS, CONTAMINATION
from src.preprocess import preprocess

FILES = {
    'healthcare': 'data/healthcare/diabetic_data.csv',
    'financial': 'data/financial/creditcard.csv',
    'realestate': 'data/realestate/train.csv',
}

MODELS_DIR = 'models'
EPOCHS = 50
BATCH_SIZE = 256
LR = 1e-3


def train_with_history(X, epochs=EPOCHS, batch_size=BATCH_SIZE, lr=LR):
    """Same architecture/training as src/autoencoder.py, but also returns
    the per-epoch loss history so we can confirm convergence."""
    X_tensor = torch.FloatTensor(X)
    loader = DataLoader(TensorDataset(X_tensor), batch_size=batch_size, shuffle=True)

    model = Autoencoder(input_dim=X.shape[1])
    optimiser = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    loss_history = []
    for epoch in range(epochs):
        total_loss = 0
        for (batch,) in loader:
            optimiser.zero_grad()
            recon = model(batch)
            loss = criterion(recon, batch)
            loss.backward()
            optimiser.step()
            total_loss += loss.item()
        avg_loss = total_loss / len(loader)
        loss_history.append(avg_loss)
        if epoch % 10 == 0 or epoch == epochs - 1:
            print(f'  Epoch {epoch:03d} | Loss: {avg_loss:.4f}')

    return model, loss_history


def train_domain(domain, path):
    print(f'\n=== {domain.upper()} ===')
    df = pd.read_csv(path)
    cfg = CONFIGS[domain]

    X, scaler, cols = preprocess(df, cfg['numeric'], cfg.get('categorical', []))

    print(f'Training on {X.shape[0]} rows, {X.shape[1]} features...')
    model, loss_history = train_with_history(X)
    model.eval()

    first_loss, last_loss = loss_history[0], loss_history[-1]
    improved = last_loss < first_loss
    print(f'Loss: {first_loss:.4f} -> {last_loss:.4f} ({"decreasing, good" if improved else "NOT decreasing -- check scaling/lr"})')

    contamination = CONTAMINATION[domain]
    threshold_percentile = 100 * (1 - contamination)

    X_tensor = torch.FloatTensor(X)
    errors = model.reconstruction_error(X_tensor)
    threshold = np.percentile(errors, threshold_percentile)
    anomalies = errors > threshold
    n_anomalies = anomalies.sum()

    print(f'Contamination: {contamination} (threshold percentile: {threshold_percentile:.2f})')
    print(f'Anomalies found: {n_anomalies} / {len(X)} ({n_anomalies / len(X):.2%})')

    top_idx = np.argsort(errors)[::-1][:20]
    display_cols = cfg['numeric'][:5]
    print(f'\nTop 20 anomalies by reconstruction error (showing {display_cols}):')
    print(df.iloc[top_idx][display_cols].to_string())

    if domain == 'financial' and 'Class' in df.columns:
        flagged = df.iloc[np.where(anomalies)[0]]
        precision = flagged['Class'].mean() if len(flagged) > 0 else 0
        recall = (flagged['Class'] == 1).sum() / df['Class'].sum()
        print(f'\nPrecision on fraud: {precision:.1%}')
        print(f'Recall on fraud: {recall:.1%}')
    else:
        print(f'\nNo ground-truth labels for {domain}. Manually review the rows above.')

    model_path = f'{MODELS_DIR}/{domain}_autoencoder.pt'
    torch.save(model.state_dict(), model_path)
    print(f'\nSaved model: {model_path}')

    return {
        'domain': domain,
        'first_loss': first_loss,
        'last_loss': last_loss,
        'n_anomalies': int(n_anomalies),
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
        print(f"  {r['domain']:12s} loss {r['first_loss']:.4f} -> {r['last_loss']:.4f}   anomalies={r['n_anomalies']}")


if __name__ == '__main__':
    main()