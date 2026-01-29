from __future__ import annotations

import json
import random
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.neighbors import LocalOutlierFactor
from sklearn.svm import OneClassSVM
from sklearn.decomposition import PCA
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import average_precision_score

from .config import NUM_COLS, TRAIN_RATIO, VAL_RATIO, IFOREST_PARAMS, RF_PARAMS
from .data import load_csv
from .eval_utils import time_sort, time_split_indices
from .run_utils import make_run_dir


SEED = 42
rng = random.Random(SEED)


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


def _prep():
    df = load_csv("data/network_dataset_labeled.csv")
    df = time_sort(df)
    df = df.drop(columns=[c for c in df.columns if c.startswith("anomaly_")], errors="ignore")
    df_feat = _build_features(df, [c for c in NUM_COLS if c in df.columns])
    y = df_feat["anomaly"].astype(int).values
    X = df_feat.drop(columns=["anomaly"], errors="ignore").select_dtypes(include=[np.number]).values
    idx_train, idx_val, idx_test = time_split_indices(len(df_feat), TRAIN_RATIO, VAL_RATIO)
    return X[idx_train], X[idx_val], y[idx_train], y[idx_val]


def _score_pr(scores, y):
    return float(average_precision_score(y, scores))


def main():
    run_dir = make_run_dir("tune")
    X_train, X_val, y_train, y_val = _prep()

    scaler = RobustScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_val_s = scaler.transform(X_val)

    results = []

    for _ in range(15):
        params = dict(IFOREST_PARAMS)
        params["n_estimators"] = rng.choice([200, 400, 800])
        params["max_samples"] = rng.choice(["auto", 0.6, 0.8])
        params["contamination"] = rng.choice([0.01, 0.03, 0.05, 0.1])
        model = IsolationForest(**params)
        model.fit(X_train_s)
        val_scores = -model.decision_function(X_val_s)
        pr = _score_pr(val_scores, y_val)
        results.append({"model": "IsolationForest", "params": params, "pr_auc": pr})

    for _ in range(10):
        params = {"n_neighbors": rng.choice([10, 20, 35]), "novelty": True}
        model = LocalOutlierFactor(**params)
        model.fit(X_train_s)
        val_scores = -model.score_samples(X_val_s)
        pr = _score_pr(val_scores, y_val)
        results.append({"model": "LOF", "params": params, "pr_auc": pr})

    for _ in range(10):
        params = {"kernel": "rbf", "gamma": rng.choice(["scale", 0.1, 0.01]), "nu": rng.choice([0.05, 0.1, 0.2])}
        model = OneClassSVM(**params)
        model.fit(X_train_s)
        val_scores = -model.decision_function(X_val_s)
        pr = _score_pr(val_scores, y_val)
        results.append({"model": "OneClassSVM", "params": params, "pr_auc": pr})

    for _ in range(10):
        n_comp = rng.choice([5, 10, 15])
        pca = PCA(n_components=min(n_comp, X_train_s.shape[1]))
        pca.fit(X_train_s)
        X_val_rec = pca.inverse_transform(pca.transform(X_val_s))
        val_scores = ((X_val_s - X_val_rec) ** 2).mean(axis=1)
        pr = _score_pr(val_scores, y_val)
        results.append({"model": "PCA_Recon", "params": {"n_components": int(pca.n_components_)}, "pr_auc": pr})

    for _ in range(10):
        params = dict(RF_PARAMS)
        params["n_estimators"] = rng.choice([200, 400, 800])
        params["max_depth"] = rng.choice([None, 6, 10])
        params["min_samples_leaf"] = rng.choice([1, 2, 4])
        model = RandomForestClassifier(**params)
        model.fit(X_train, y_train)
        val_scores = model.predict_proba(X_val)[:, 1]
        pr = _score_pr(val_scores, y_val)
        results.append({"model": "RandomForest", "params": params, "pr_auc": pr})

    results = sorted(results, key=lambda x: x["pr_auc"], reverse=True)
    (run_dir / "results.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    (run_dir / "best_params.json").write_text(json.dumps(results[0], indent=2), encoding="utf-8")
    print(f"OK Tuning done. Saved {run_dir}")
    print(results[0])


if __name__ == "__main__":
    main()
