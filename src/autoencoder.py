"""
PyTorch Autoencoder for anomaly detection via reconstruction error.

Trained to reconstruct normal data. Anomalies produce high reconstruction
error because the model never learned to represent them.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset


class Autoencoder(nn.Module):
    def __init__(self, input_dim, encoding_dim=16):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, encoding_dim),
            nn.ReLU()
        )
        self.decoder = nn.Sequential(
            nn.Linear(encoding_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 64),
            nn.ReLU(),
            nn.Linear(64, input_dim)
        )

    def forward(self, x):
        return self.decoder(self.encoder(x))

    def reconstruction_error(self, x):
        """MSE per sample. Higher = more anomalous."""
        with torch.no_grad():
            recon = self.forward(x)
        return ((x - recon) ** 2).mean(dim=1).numpy()


def train_autoencoder(X, epochs=50, batch_size=256, lr=1e-3):
    X_tensor = torch.FloatTensor(X)
    loader = DataLoader(TensorDataset(X_tensor), batch_size=batch_size, shuffle=True)

    model = Autoencoder(input_dim=X.shape[1])
    optimiser = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    for epoch in range(epochs):
        total_loss = 0
        for (batch,) in loader:
            optimiser.zero_grad()
            recon = model(batch)
            loss = criterion(recon, batch)
            loss.backward()
            optimiser.step()
            total_loss += loss.item()
        if epoch % 10 == 0:
            print(f'Epoch {epoch:03d} | Loss: {total_loss / len(loader):.4f}')

    return model


def score_with_autoencoder(model, X, threshold_percentile=95):
    X_tensor = torch.FloatTensor(X)
    errors = model.reconstruction_error(X_tensor)
    threshold = np.percentile(errors, threshold_percentile)
    anomalies = errors > threshold
    return errors, anomalies
