"""
Optional FastAPI backend for programmatic access.

Run with: uvicorn app.api:app --reload
"""

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
from fastapi import FastAPI, UploadFile, File
from io import StringIO

from src.config import CONFIGS
from src.isolation import IsolationDetector
from src.preprocess import preprocess

MODELS_DIR = Path(__file__).resolve().parent.parent / 'models'

app = FastAPI(title='Anomaly Detection Engine API')


@app.post('/detect')
async def detect(domain: str, top_n: int = 20, file: UploadFile = File(...)):
    """
    Upload a CSV for the given domain and get back the top N anomalous rows
    (by Isolation Forest score) as JSON.
    """
    if domain not in CONFIGS:
        return {'error': f'Unknown domain "{domain}". Choose from {list(CONFIGS.keys())}.'}

    contents = await file.read()
    df = pd.read_csv(StringIO(contents.decode('utf-8')))
    cfg = CONFIGS[domain]

    X, scaler, cols = preprocess(df, cfg['numeric'], cfg.get('categorical', []))

    model_path = MODELS_DIR / f'{domain}_isolation.joblib'
    if not model_path.exists():
        return {'error': f'No trained model found for domain "{domain}". Train it first.'}

    detector = IsolationDetector().load(str(model_path))
    scores = detector.score(X)
    top_idx = np.argsort(scores)[:top_n]

    return {
        'domain': domain,
        'n_rows': len(df),
        'anomalies': df.iloc[top_idx].to_dict(orient='records'),
        'scores': scores[top_idx].tolist(),
    }
