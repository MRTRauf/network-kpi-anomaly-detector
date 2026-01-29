from __future__ import annotations

import json
import os
import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_fscore_support, roc_auc_score, average_precision_score

from .config import NUM_COLS, INCIDENT_GAP_ROWS, TRAIN_RATIO, VAL_RATIO
from .data import load_csv
from .features import add_time_series_features, get_feature_columns, per_row_contribution_z
from .incidents import group_incidents
from .eval_utils import time_sort, time_split_indices, precision_recall_at_k
from .run_utils import make_run_dir, latest_run_dir

ART_DIR = "artifacts"


def _time_split(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    n = len(df)
    train_end = int(n * TRAIN_RATIO)
    val_end = int(n * (TRAIN_RATIO + VAL_RATIO))
    train_df = df.iloc[:train_end].copy()
    val_df = df.iloc[train_end:val_end].copy()
    test_df = df.iloc[val_end:].copy()
    return train_df, val_df, test_df


def main():
    train_dir = latest_run_dir("train")
    if train_dir is None:
        raise RuntimeError("No training run found under artifacts/run_train_*. Run: python -m src.train")

    meta = joblib.load(train_dir / "meta.joblib")
    thr = float(meta["threshold"])
    model_type = meta.get("model_type", "IsolationForest")

    df = load_csv("data/network_dataset_labeled.csv")
    if "anomaly" not in df.columns:
        raise ValueError("Expected 'anomaly' label column in labeled dataset.")

    df = time_sort(df)
    df_feat = add_time_series_features(df, NUM_COLS)
    df_feat = per_row_contribution_z(df_feat, NUM_COLS)

    model = joblib.load(train_dir / "model.joblib")
    scaler = joblib.load(train_dir / "scaler.joblib")

    feat_cols = meta.get("feature_cols", get_feature_columns(NUM_COLS))
    X = df_feat[feat_cols].astype(float).values

    if model_type == "IsolationForest":
        Xs = scaler.transform(X)
        score = -model.decision_function(Xs)
    else:
        score = model.predict_proba(X)[:, 1]

    df_feat["anomaly_score"] = score
    df_feat["is_alert"] = df_feat["anomaly_score"] >= thr

    y_true = df_feat["anomaly"].astype(int).values
    y_pred = df_feat["is_alert"].astype(int).values

    idx_train, idx_val, idx_test = time_split_indices(len(df_feat), TRAIN_RATIO, VAL_RATIO)
    test_mask = idx_test

    y_test = y_true[test_mask]
    score_test = score[test_mask]
    pred_test = y_pred[test_mask]

    p, r, f1, _ = precision_recall_fscore_support(y_test, pred_test, average="binary", zero_division=0)

    try:
        roc_auc = roc_auc_score(y_test, score_test)
    except Exception:
        roc_auc = float("nan")

    try:
        pr_auc = average_precision_score(y_test, score_test)
    except Exception:
        pr_auc = float("nan")

    metrics = {
        "model": model_type,
        "threshold": thr,
        "precision": float(p),
        "recall": float(r),
        "f1": float(f1),
        "roc_auc": float(roc_auc),
        "pr_auc": float(pr_auc),
        "positive_rate_true": float(np.mean(y_true)),
        "alert_rate_pred": float(np.mean(y_pred)),
    }
    metrics.update(precision_recall_at_k(y_test, score_test, ks=[0.01, 0.02, 0.05, 0.1]))

    run_dir = make_run_dir("eval")
    pd.DataFrame([metrics]).to_csv(run_dir / "eval_metrics.csv", index=False)
    (run_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    try:
        df_feat.to_parquet(run_dir / "scored_labeled.parquet", index=False)
    except Exception:
        df_feat.to_csv(run_dir / "scored_labeled.csv", index=False)

    incidents = group_incidents(df_feat, gap_rows=INCIDENT_GAP_ROWS)
    try:
        incidents.to_parquet(run_dir / "incidents_labeled.parquet", index=False)
    except Exception:
        incidents.to_csv(run_dir / "incidents_labeled.csv", index=False)

    print(f"OK Evaluation done. Saved {run_dir}/eval_metrics.csv")
    print(metrics)


if __name__ == "__main__":
    main()
