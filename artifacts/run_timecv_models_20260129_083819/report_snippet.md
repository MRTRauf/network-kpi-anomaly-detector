# Time-CV model comparison

- data_path: data/network_dataset_labeled.csv
- leakage controls: drop anomaly_* and target-like columns
- folds_used: 5
- best_model: GradientBoosting

## Summary (mean ± std)
| model | precision | recall | f1 | f2 | pr_auc |
|---|---:|---:|---:|---:|---:|
| LogisticRegression | 0.2835 ± 0.1706 | 0.6122 ± 0.3180 | 0.3737 ± 0.2002 | 0.4781 ± 0.2405 | 0.6632 ± 0.0173 |
| RandomForest | 0.3815 ± 0.1536 | 0.4178 ± 0.2963 | 0.3251 ± 0.0456 | 0.3459 ± 0.1271 | 0.4806 ± 0.0317 |
| GradientBoosting | 0.6325 ± 0.1319 | 0.8133 ± 0.2225 | 0.6784 ± 0.1201 | 0.7422 ± 0.1593 | 0.8592 ± 0.0511 |