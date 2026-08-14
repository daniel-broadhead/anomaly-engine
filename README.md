# Anomaly Detection Engine

A configurable anomaly detection pipeline demonstrating Isolation Forest and
Autoencoder methods across healthcare, financial, and real estate domains.

<!-- Demo GIF goes here once Phase 6 is complete -->

## Architecture

```
Data -> Preprocessing -> [Isolation Forest | Autoencoder] -> SHAP / Reconstruction-Error Explainer -> Streamlit UI
```

## Results

<!-- Fill in after Phase 3/4:
- Precision on the fraud dataset (target: >30%)
- Qualitative validation notes for healthcare and real estate
-->

## Design decisions

<!-- Fill in as you build -- this section is what separates a career-level
README from an intern README. Cover:
- Why two methods (Isolation Forest + Autoencoder)?
- Why these three domains?
- How was the contamination threshold chosen per domain?
- Per-feature reconstruction error attribution vs. true SHAP -- why, and
  where the honesty matters
-->

## Limitations & next steps

<!-- e.g. semi-supervised approach, streaming data, retraining API, VAE -->

## Running locally

```bash
python -m venv venv
source venv/bin/activate    # Windows: venv\Scripts\activate
pip install -r requirements.txt

# 1. Download the three datasets into data/<domain>/ (see project guide)
# 2. Train models for each domain (Phase 3 & 4)
# 3. Launch the app
streamlit run app/streamlit_app.py
```

## Tech stack

scikit-learn, PyTorch, SHAP, pandas/NumPy, Streamlit, FastAPI, Docker.
