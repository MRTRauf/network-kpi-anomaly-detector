from __future__ import annotations

import json
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import average_precision_score

from .config import NUM_COLS, TRAIN_RATIO, VAL_RATIO, IFOREST_PARAMS
from .data import load_csv
from .eval_utils import time_sort, time_split_indices
from .run_utils import make_run_dir


def _build_features(df: pd.DataFrame, num_cols: list[str]) -> pd.DataFrame:
    out = df.copy()
    for c in num_cols:
        out[c] = pd.to_numeric(out[c], errors="coerce").interpolate(limit_direction="both")
    for c in num_cols:
        out[f"{c}__diff1"] = out[c].diff().fillna(0.0)
        out[f"{c}__roll5_mean"] = out[c].rolling(5, min_periods=1).mean()
        out[f"{c}__roll5_std"] = out[c].rolling(5, min_periods=1).std().fillna(0.0)
    out = out.dropna().reset_index(drop=True)
    return out


def _score_if(X_train: np.ndarray, X_test: np.ndarray) -> np.ndarray:
    scaler = RobustScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)
    model = IsolationForest(**IFOREST_PARAMS)
    model.fit(X_train_s)
    return -model.decision_function(X_test_s)


def main():
    run_dir = make_run_dir("sanity")
    df = load_csv("data/network_dataset_labeled.csv")
    if "anomaly" not in df.columns:
        raise ValueError("Expected 'anomaly' label column in labeled dataset.")

    df = time_sort(df)
    leakage_cols = [c for c in df.columns if c.startswith("anomaly_")]
    df = df.drop(columns=leakage_cols, errors="ignore")

    feature_cols = [c for c in NUM_COLS if c in df.columns]
    df_feat = _build_features(df, feature_cols)

    y = df_feat["anomaly"].astype(int).values
    X = df_feat.drop(columns=["anomaly"], errors="ignore").select_dtypes(include=[np.number]).values

    idx_train, idx_val, idx_test = time_split_indices(len(df_feat), TRAIN_RATIO, VAL_RATIO)
    X_train, X_test = X[idx_train], X[idx_test]
    y_train, y_test = y[idx_train], y[idx_test]

    base_scores = _score_if(X_train, X_test)
    base_pr = average_precision_score(y_test, base_scores)

    y_test_rand = np.random.RandomState(42).permutation(y_test)
    rand_pr = average_precision_score(y_test_rand, base_scores)

    rng = np.random.RandomState(42)
    X_test_shuf = X_test.copy()
    for j in range(X_test_shuf.shape[1]):
        rng.shuffle(X_test_shuf[:, j])
    shuf_scores = _score_if(X_train, X_test_shuf)
    shuf_pr = average_precision_score(y_test, shuf_scores)

    X_test_shift = np.roll(X_test, shift=1, axis=0)
    shift_scores = _score_if(X_train, X_test_shift)
    shift_pr = average_precision_score(y_test, shift_scores)

    out = {
        "baseline_pr_auc": float(base_pr),
        "random_labels_pr_auc": float(rand_pr),
        "shuffled_features_pr_auc": float(shuf_pr),
        "time_shift_pr_auc": float(shift_pr),
    }

    (run_dir / "metrics.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"OK Sanity checks done. Saved {run_dir}")
    print(out)


if __name__ == "__main__":
    main()
