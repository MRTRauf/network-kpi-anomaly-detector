# network-kpi-anomaly-detector

Detects anomalies in network KPI time series so NOC teams catch incidents early with fewer missed anomalies.

Highlights
- Leakage-safe feature set with time-aware validation.
- Thresholds tuned for recall/F2 to reduce false negatives.
- Dashboard and API for triage and scoring.

Quickstart (3 commands)
```bash
pip install -r requirements.txt
python -m src.optimize
streamlit run src/dashboard.py
```

Data
Place datasets in `data/`:
- `data/network_dataset.csv`
- `data/network_dataset_labeled.csv`

Results (rolling time-CV, tuned Gradient Boosting)
Best config: n_estimators=100, learning_rate=0.1, max_depth=2, min_samples_leaf=5, subsample=0.85.

| Metric | Mean +/- Std |
|---|---|
| Recall | 0.9356 +/- 0.0528 |
| F2 | 0.8928 +/- 0.0871 |
| F1 | 0.8447 +/- 0.1319 |
| Precision | 0.7868 +/- 0.1762 |
| PR-AUC | 0.9546 +/- 0.0373 |
| Alert rate | 0.0880 +/- 0.0204 |

Operating points (test aggregate)
- F2-opt: threshold=0.0255, recall=0.9434, precision=0.4762, alerts≈140 per 1000
- High-recall: threshold=0.0352, recall=0.9434, precision=0.5102, alerts≈131 per 1000
- Alert-budget (5%): threshold=0.8509, recall=0.3962, precision=1.0000, alerts≈28 per 1000

We benchmarked 3 model families; Gradient Boosting performed best under rolling time-CV.

How it avoids leakage
- Drops all `anomaly_*` columns and target-like columns at train time.
- Uses chronological splits and rolling time-CV.
- Rolling features are past-only (lag and rolling stats use prior rows).

Notebooks
- `notebooks/Quickstart.ipynb`: End-to-end run and tuned results summary.
- `notebooks/EDA.ipynb`: Data overview, anomaly behavior, and drift checks.
- `notebooks/Model_Comparison.ipynb`: Model benchmark, tuning table, and operating points.

Reproduce tuning
```bash
python -m src.timecv_tune_gb
```

API (optional)
```bash
uvicorn src.api:app --reload
```
