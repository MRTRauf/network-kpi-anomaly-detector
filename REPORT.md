# Technical Report

## Dataset summary
- Source: `data/network_dataset_labeled.csv`
- Rows: 1001
- Label rate (anomaly=1): 0.0829
- Timestamp range: 2024-05-11 12:00:15+00:00 to 2024-05-11 23:26:43+00:00

## Evaluation protocol
- Rolling/expanding time-series cross-validation (no shuffling).
- 5 folds, each with a validation segment for threshold tuning and a test segment for evaluation.
- Threshold chosen per fold to maximize F2 on validation.
- Metrics reported on the test segment.
- Leakage controls: drop anomaly_* and target-like columns.

## Model comparison (rolling time-CV, mean +/- std)
| Model | Precision | Recall | F1 | F2 | PR-AUC |
|---|---:|---:|---:|---:|---:|
| LogisticRegression | 0.2835 +/- 0.1706 | 0.6122 +/- 0.3180 | 0.3737 +/- 0.2002 | 0.4781 +/- 0.2405 | 0.6632 +/- 0.0173 |
| RandomForest | 0.3815 +/- 0.1536 | 0.4178 +/- 0.2963 | 0.3251 +/- 0.0456 | 0.3459 +/- 0.1271 | 0.4806 +/- 0.0317 |
| GradientBoosting | 0.6325 +/- 0.1319 | 0.8133 +/- 0.2225 | 0.6784 +/- 0.1201 | 0.7422 +/- 0.1593 | 0.8592 +/- 0.0511 |

## GradientBoosting tuning (rolling time-CV)
Best config
- n_estimators=100, learning_rate=0.1, max_depth=2, min_samples_leaf=5, subsample=0.85

Tuned metrics (mean +/- std)
- Precision: 0.7868 +/- 0.1762
- Recall: 0.9356 +/- 0.0528
- F1: 0.8447 +/- 0.1319
- F2: 0.8928 +/- 0.0871
- PR-AUC: 0.9546 +/- 0.0373
- Alert rate: 0.0880 +/- 0.0204

Operating points (test aggregate)
- F2-opt: threshold=0.0255, recall=0.9434, precision=0.4762, F2=0.7886, alerts≈140 per 1000
- High-recall: threshold=0.0352, recall=0.9434, precision=0.5102, F2=0.8065, alerts≈131 per 1000
- Alert-budget (5%): threshold=0.8509, recall=0.3962, precision=1.0000, F2=0.4506, alerts≈28 per 1000

Feature importance (top 5, mean)
- congestion
- packet_loss
- latency
- jitter
- throughput

Evidence runs
- `artifacts/run_timecv_models_20260129_083819/`
- `artifacts/run_timecv_tune_gb_20260129_090231/`

## Notebook walkthrough
- Quickstart runs the pipeline end-to-end and summarizes tuned operating points.
- EDA focuses on anomaly behavior, missingness, and time trends.
- Model Comparison shows the 3-model benchmark, tuning table, and explainability.
