# Anomaly Detection Engine

A configurable anomaly detection pipeline demonstrating Isolation Forest and
Autoencoder methods across healthcare, financial, and real estate domains.

## Architecture

```
Data -> Preprocessing -> [Isolation Forest | Autoencoder] -> SHAP / Reconstruction-Error Explainer -> Streamlit UI
```

## Results

| Domain | Method | Result |
|---|---|---|
| Financial (fraud) | Isolation Forest | 26.3% precision, ~30.5% recall against 492 known fraud cases (contamination=0.002, matched to the true 0.173% fraud rate) |
| Financial (fraud) | Autoencoder | 10.5% precision, 12.2% recall |
| Financial (fraud) | Union of both methods | 16.3% precision, 32.7% recall — a small recall gain at a real precision cost (see Design Decisions) |
| Healthcare | Isolation Forest | 5,089 rows flagged (5% contamination); manually validated for domain plausibility, no ground-truth labels available |
| Healthcare | Autoencoder | 5,089 rows flagged; top flags driven by elevated lab-procedure counts and length of stay |
| Real estate | Isolation Forest | 73 rows flagged (5% contamination) |
| Real estate | Autoencoder | 73 rows flagged; top flags include a 5,642 sq ft living area and a $745,000 sale price |

**Cross-method agreement** (Jaccard similarity of flagged rows): 22.7% (healthcare), 15.7% (financial), 16.8% (real estate). The two methods agree on only about a fifth to a quarter of what they flag across every domain — they are finding largely different anomalies, not the same ones through two lenses.

## Design decisions

### Why two methods?

Isolation Forest and a PyTorch Autoencoder are complementary rather than redundant. Isolation Forest requires no training beyond fitting on the data distribution, runs in seconds even on hundreds of thousands of rows, and performs well with almost no tuning. The Autoencoder is slower to train and more sensitive to preprocessing, but can capture non-linear relationships between features that axis-aligned tree splits miss. Offering both lets a user trade speed and interpretability against the ability to catch subtler patterns.

That trade-off isn't just theoretical — it's measurable. On the financial (fraud) domain, where ground-truth labels exist, Isolation Forest alone reaches 26.3% precision and ~30.5% recall, while the Autoencoder alone reaches 10.5% precision and 12.2% recall. Isolation Forest is doing most of the useful work here. Combining both methods (flagging anything either one catches) only lifts recall to 32.7% — a 2-point gain — while precision drops to 16.3%, since nearly 400 additional rows now need review to find 11 extra true fraud cases. The honest conclusion isn't "two methods beat one"; it's that Isolation Forest should likely be the primary method for this dataset, with the Autoencoder contributing a small number of unique catches. Whether that trade is worth it in production depends on the relative cost of a missed fraud case versus a human reviewing a false positive.

### Why these three domains?

Healthcare, financial, and real estate were chosen to demonstrate that the same architecture generalizes across genuinely different data types (clinical records, anonymized PCA-transformed transaction features, and structured property listings) rather than being tuned to one problem. Each domain also connects to direct prior experience: pharmacy/biochemistry background for the clinical data, hands-on work with international real estate databases, and high-volume data validation experience from prior data/quality roles.

### Contamination threshold choices

Both methods need an estimate of what fraction of the data is anomalous. For the financial dataset, the true fraud rate is known (492 of 284,807 transactions, ~0.173%), so contamination was set close to that ground truth (0.002). Healthcare and real estate have no ground truth, so contamination uses an exploratory default of 5%, validated by manually reviewing the top-flagged rows for domain plausibility rather than any statistical target.

### A preprocessing bug the explainability layer caught

During the explainability phase, SHAP and reconstruction-error attributions for both methods were dominated, in every domain with categorical features, by one-hot encoded columns representing *rare* categories (an uncommon race/gender/age bracket in healthcare, a rare neighborhood in real estate) rather than genuine numeric outliers. The root cause: `StandardScaler` was being applied uniformly to both continuous numeric columns and one-hot dummy columns. For a rare category present in only a handful of rows, the column's standard deviation is tiny, so scaling inflated its value for those rows to 10, 50, even 180+ standard deviations — a number with no real meaning, since a 0/1 presence indicator doesn't have a magnitude to standardize in the first place.

The fix: scale only the genuinely numeric columns, leave one-hot dummy columns as raw 0/1. After retraining, the Autoencoder's flagged anomalies shifted meaningfully toward real signal. For real estate, the previously dominant "rare neighborhood" artifact disappeared entirely, replaced by genuine outliers like a 5,642 sq ft living area and a $745,000 sale price. For healthcare, per-row reconstruction error dropped by roughly two orders of magnitude, and numeric features like lab-procedure counts and length of stay began surfacing as top contributors for the first time.

Isolation Forest's flagged rows, by contrast, barely changed after the fix. Isolation Forest's random-partition splits are invariant to monotonic rescaling of a single feature — whether a rare category is encoded as `0/1` or `0/180`, a random split threshold between the two values partitions the data identically. Its tendency to flag rare categories was never actually a scaling artifact; it's a structural property of the algorithm, since a small, distinct subset is inherently easy to isolate via random partitioning regardless of numeric encoding. The Autoencoder, which optimizes squared reconstruction error, is not scale-invariant in the same way — which is exactly why the bug affected one method and not the other, and a useful illustration of how these two algorithm families actually differ under the hood.

### Per-feature reconstruction error attribution vs. SHAP

Autoencoders have no native SHAP explainer, so anomaly explanations for that method use per-feature squared reconstruction error instead: whichever features the model reconstructed worst for a given row are reported as the top contributors. This is *not* SHAP, and is described accurately throughout this project as "feature-level reconstruction error attribution" rather than implied to be SHAP under a different name. It's interpretable, cheap to compute, and honest about what it measures — a true Shapley-value-based explanation for a neural network would require a different, more expensive approach (e.g. KernelExplainer) that wasn't necessary at this project's scale.

### Reproducibility

Autoencoder training initially had no fixed random seed, so PyTorch's default weight initialization and the training DataLoader's batch shuffling both varied between runs — the same data could produce different flagged anomalies on different runs. This was caught by noticing the financial domain's explained anomalies changed between two runs despite financial having no categorical columns (and therefore being unaffected by the preprocessing fix above) — the only remaining source of variation was an unseeded RNG. `torch.manual_seed(42)`, matching Isolation Forest's own `random_state=42`, is now set at the start of every training run, and produces identical loss curves and identical flagged rows across repeated runs on the same data.

## Limitations and next steps

- Stability checks (retraining with *different* random seeds and confirming the flagged rows stay roughly consistent) haven't been run for any domain yet. This is distinct from the reproducibility fix above, which only guarantees the *same* seed reproduces the *same* result — it says nothing about how sensitive the flagged anomalies are to the seed choice itself. Worth adding, especially for healthcare and real estate, since they have no ground-truth labels to validate against any other way.
- The financial dataset's duplicate transaction rows haven't been deduplicated before training; worth checking whether this affects flagged results.
- A Variational Autoencoder (VAE) would give proper probability estimates on anomaly scores instead of raw reconstruction error — a natural next step if pursuing this further.
- Production deployment would need a retraining/drift-detection pipeline, an authenticated API, and a human-in-the-loop review queue rather than automated action on flagged anomalies.

## Running locally

Trained models are included in this repo (~7MB total), so the app runs immediately after cloning — no dataset download or training step required:

```bash
git clone <this-repo-url>
cd anomaly-engine
python -m venv venv
source venv/bin/activate    # Windows: venv\Scripts\activate
pip install -r requirements.txt
streamlit run app/streamlit_app.py
```

Upload any CSV matching one of the three domain schemas (see `src/config.py`) to try it out.

**To retrain the models yourself** (optional — only needed if you want to reproduce or modify the pipeline):

1. Download the three datasets from Kaggle: "diabetes 130 hospitals" (healthcare), "credit card fraud detection" (financial), "house prices ames iowa" (real estate). Place each CSV in its matching `data/<domain>/` folder.
2. Run the training scripts:
   ```bash
   python scripts/train_isolation.py
   python scripts/train_autoencoder.py
   ```

## Tech stack

scikit-learn, PyTorch, SHAP, pandas/NumPy, Streamlit, Docker.