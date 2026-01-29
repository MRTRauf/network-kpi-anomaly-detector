from __future__ import annotations

import json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.svm import OneClassSVM
from sklearn.decomposition import PCA
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import precision_recall_fscore_support, roc_auc_score, average_precision_score

from .config import NUM_COLS, TRAIN_RATIO, VAL_RATIO, IFOREST_PARAMS
from .data import load_csv
from .eval_utils import time_sort, time_split_indices, precision_recall_at_k
from .run_utils import make_run_dir


def _build_features(df: pd.DataFrame, num_cols: list[str]) -> pd.DataFrame:
    out = df.copy()
    for c in num_cols:
        out[c] = pd.to_numeric(out[c], errors="coerce").interpolate(limit_direction="both")
    lags = [1, 2, 3]
    windows = [5, 10]
    for c in num_cols:
        for l in lags:
            out[f"{c}__lag{l}"] = out[c].shift(l)
        for w in windows:
            roll = out[c].rolling(w, min_periods=1)
            out[f"{c}__roll{w}_mean"] = roll.mean()
            out[f"{c}__roll{w}_std"] = roll.std().fillna(0.0)
            out[f"{c}__roll{w}_median"] = roll.median()
            out[f"{c}__roll{w}_q10"] = roll.quantile(0.10)
            out[f"{c}__roll{w}_q90"] = roll.quantile(0.90)
        out[f"{c}__diff1"] = out[c].diff().fillna(0.0)
    out = out.dropna().reset_index(drop=True)
    return out


def _drop_constant(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    nunique = df.nunique()
    const = nunique[nunique <= 1].index.tolist()
    return df.drop(columns=const, errors="ignore"), const


def _tune_threshold(scores: np.ndarray, y_true: np.ndarray) -> float:
    qs = np.linspace(0.80, 0.995, 50)
    thresholds = np.quantile(scores, qs)
    best_thr = float(thresholds[0])
    best_f1 = -1.0
    for thr in thresholds:
        y_pred = (scores >= thr).astype(int)
        _, _, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="binary", zero_division=0)
        if f1 > best_f1:
            best_f1 = float(f1)
            best_thr = float(thr)
    return best_thr


def _eval_model(name: str, scores_val: np.ndarray, scores_test: np.ndarray, y_val: np.ndarray, y_test: np.ndarray) -> dict:
    thr = _tune_threshold(scores_val, y_val)
    y_pred = (scores_test >= thr).astype(int)
    p, r, f1, _ = precision_recall_fscore_support(y_test, y_pred, average="binary", zero_division=0)
    roc = roc_auc_score(y_test, scores_test)
    pr = average_precision_score(y_test, scores_test)
    metrics = {
        "model": name,
        "threshold": float(thr),
        "precision": float(p),
        "recall": float(r),
        "f1": float(f1),
        "roc_auc": float(roc),
        "pr_auc": float(pr),
    }
    metrics.update(precision_recall_at_k(y_test, scores_test, ks=[0.01, 0.02, 0.05, 0.1]))
    return metrics


def main():
    run_dir = make_run_dir("benchmark")
    df = load_csv("data/network_dataset_labeled.csv")
    if "anomaly" not in df.columns:
        raise ValueError("Expected 'anomaly' label column in labeled dataset.")

    df = time_sort(df)

    leakage_cols = [c for c in df.columns if c.startswith("anomaly_")]
    df = df.drop(columns=leakage_cols, errors="ignore")

    feature_cols = [c for c in NUM_COLS if c in df.columns]
    df_feat = _build_features(df, feature_cols)
    df_feat, const_cols = _drop_constant(df_feat)

    y = df_feat["anomaly"].astype(int).values
    X_df = df_feat.drop(columns=["anomaly"], errors="ignore")
    X_df = X_df.select_dtypes(include=[np.number])
    X = X_df.values

    idx_train, idx_val, idx_test = time_split_indices(len(df_feat), TRAIN_RATIO, VAL_RATIO)
    X_train, X_val, X_test = X[idx_train], X[idx_val], X[idx_test]
    y_train, y_val, y_test = y[idx_train], y[idx_val], y[idx_test]

    scaler = RobustScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_val_s = scaler.transform(X_val)
    X_test_s = scaler.transform(X_test)

    results = []
    params = {}

    if_params = dict(IFOREST_PARAMS)
    if_model = IsolationForest(**if_params)
    if_model.fit(X_train_s)
    if_val = -if_model.decision_function(X_val_s)
    if_test = -if_model.decision_function(X_test_s)
    results.append(_eval_model("IsolationForest", if_val, if_test, y_val, y_test))
    params["IsolationForest"] = if_params

    lof = LocalOutlierFactor(n_neighbors=20, contamination="auto", novelty=True)
    lof.fit(X_train_s)
    lof_val = -lof.score_samples(X_val_s)
    lof_test = -lof.score_samples(X_test_s)
    results.append(_eval_model("LOF", lof_val, lof_test, y_val, y_test))
    params["LOF"] = {"n_neighbors": 20, "novelty": True}

    ocsvm = OneClassSVM(kernel="rbf", gamma="scale", nu=0.1)
    ocsvm.fit(X_train_s)
    svm_val = -ocsvm.decision_function(X_val_s)
    svm_test = -ocsvm.decision_function(X_test_s)
    results.append(_eval_model("OneClassSVM", svm_val, svm_test, y_val, y_test))
    params["OneClassSVM"] = {"kernel": "rbf", "gamma": "scale", "nu": 0.1}

    pca = PCA(n_components=min(10, X_train_s.shape[1]))
    pca.fit(X_train_s)
    X_val_rec = pca.inverse_transform(pca.transform(X_val_s))
    X_test_rec = pca.inverse_transform(pca.transform(X_test_s))
    pca_val = ((X_val_s - X_val_rec) ** 2).mean(axis=1)
    pca_test = ((X_test_s - X_test_rec) ** 2).mean(axis=1)
    results.append(_eval_model("PCA_Recon", pca_val, pca_test, y_val, y_test))
    params["PCA_Recon"] = {"n_components": int(pca.n_components_)}

    out = pd.DataFrame(results).sort_values(by=["pr_auc", "f1"], ascending=False)
    out.to_csv(run_dir / "metrics.csv", index=False)
    (run_dir / "metrics.json").write_text(out.to_json(orient="records", indent=2), encoding="utf-8")
    (run_dir / "best_params.json").write_text(json.dumps(params, indent=2), encoding="utf-8")

    print(f"OK Benchmark done. Saved {run_dir}")
    print(out)


if __name__ == "__main__":
    main()
