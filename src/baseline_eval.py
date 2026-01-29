from __future__ import annotations

import json
import numpy as np
import pandas as pd
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import precision_recall_fscore_support, roc_auc_score, average_precision_score

from .config import NUM_COLS, TRAIN_RATIO, VAL_RATIO
from .data import load_csv
from .eval_utils import time_sort, time_split_indices, precision_recall_at_k
from .run_utils import make_run_dir


def main():
    run_dir = make_run_dir("baseline")

    df = load_csv("data/network_dataset_labeled.csv")
    if "anomaly" not in df.columns:
        raise ValueError("Expected 'anomaly' label column in labeled dataset.")

    df = time_sort(df)

    feature_cols = [c for c in NUM_COLS if c in df.columns]
    X = df[feature_cols].astype(float).values
    y = df["anomaly"].astype(int).values

    idx_train, idx_val, idx_test = time_split_indices(len(df), TRAIN_RATIO, VAL_RATIO)
    X_train, X_test = X[idx_train], X[idx_test]
    y_train, y_test = y[idx_train], y[idx_test]

    scaler = RobustScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    train_scores = np.linalg.norm(X_train_s, axis=1)
    test_scores = np.linalg.norm(X_test_s, axis=1)

    threshold = float(np.quantile(train_scores, 0.99))
    y_pred = (test_scores >= threshold).astype(int)

    p, r, f1, _ = precision_recall_fscore_support(y_test, y_pred, average="binary", zero_division=0)
    roc = roc_auc_score(y_test, test_scores)
    pr = average_precision_score(y_test, test_scores)

    metrics = {
        "model": "baseline_l2_robust",
        "threshold": threshold,
        "precision": float(p),
        "recall": float(r),
        "f1": float(f1),
        "roc_auc": float(roc),
        "pr_auc": float(pr),
    }
    metrics.update(precision_recall_at_k(y_test, test_scores, ks=[0.01, 0.02, 0.05, 0.1]))

    scored = df.iloc[idx_test].copy()
    scored["anomaly_score"] = test_scores
    scored["is_alert"] = y_pred

    (run_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    scored.to_csv(run_dir / "scored_labeled.csv", index=False)

    print(f"OK Baseline eval done. Saved {run_dir}")
    print(metrics)


if __name__ == "__main__":
    main()
