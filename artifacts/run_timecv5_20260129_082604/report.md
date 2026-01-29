# Rolling time-CV report (5 folds max)

- data_path: data/network_dataset_labeled.csv
- feature_set: A (no anomaly_* and no target-like columns)
- model: GradientBoostingClassifier
- folds_used: 5
- min_test_anoms: 5

## Summary (mean ± std)
- precision: 0.6325 ± 0.1319
- recall: 0.8133 ± 0.2225
- f1: 0.6784 ± 0.1201
- f2: 0.7422 ± 0.1593
- pr_auc: 0.8592 ± 0.0511

See metrics_per_fold.csv for per-fold details and plots per fold.