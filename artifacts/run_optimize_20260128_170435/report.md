# Optimization report

## Data audit
- data_path: data/network_dataset_labeled.csv
- rows: 1001
- cols: 21
- anomaly_rate: 0.0829
- object_cols: ['timestamp', 'Routers', 'Planned route', 'Network measure', 'Network target', 'Video target']
- leakage_cols_removed: ['anomaly_throughput', 'anomaly_congestion', 'anomaly_packet_loss', 'anomaly_latency', 'anomaly_jitter']
- target_like_removed: ['Network target', 'Video target']

## Models tried
| model | status | threshold | test_recall | test_f2 | test_f1 | test_pr_auc |
|---|---|---:|---:|---:|---:|---:|
| LogisticRegression | ok | 0.9995 | 0.0000 | 0.0000 | 0.0000 | 0.6875 |
| RandomForest | ok | 0.0725 | 1.0000 | 0.5804 | 0.3562 | 0.5190 |
| HistGradientBoosting | failed | - | - | - | - | - |
| GradientBoosting_fallback | ok | 0.0125 | 1.0000 | 0.8784 | 0.7429 | 0.9329 |

## Best model
- best_model: GradientBoosting_fallback
- threshold (val max F2): 0.0125
- test metrics: {'precision': 0.5909090909090909, 'recall': 1.0, 'f1': 0.7428571428571429, 'f2': 0.8783783783783784, 'pr_auc': 0.9329025225786358}
- operating_point_recall>=0.70: {'threshold': 0.012482177672164458, 'precision': 0.5, 'recall': 1.0, 'f2': 0.8333333333333334}

## Sanity checks
- pr_auc_shuffled_labels: 0.1595
- pr_auc_shuffled_rows: 0.1096

## Feature sets
- A: excludes anomaly_* and target-like columns (used).
- B: anomaly_* not included due to leakage risk (skipped).