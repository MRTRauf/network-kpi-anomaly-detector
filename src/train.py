from __future__ import annotations

import json
import os
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest, RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.metrics import precision_recall_fscore_support, average_precision_score, roc_auc_score
from sklearn.preprocessing import StandardScaler

from .config import (
    NUM_COLS,
    ALERT_QUANTILE,
    IFOREST_PARAMS,
    RF_PARAMS,
    HGB_PARAMS,
    INCIDENT_GAP_ROWS,
    TRAIN_RATIO,
    VAL_RATIO,
    CV_FOLDS,
)
from .data import load_csv
from .features import add_time_series_features, get_feature_columns, per_row_contribution_z
from .incidents import group_incidents
from .eval_utils import time_sort, time_split_indices
from .run_utils import make_run_dir


def _sample_weight(y: np.ndarray) -> np.ndarray | None:
    pos = float(y.sum())
    if pos <= 0:
        return None
    neg = float(len(y) - pos)
    if neg <= 0:
        return None
    w = neg / pos
    return np.where(y == 1, w, 1.0)


def _tune_threshold(scores: np.ndarray, y_true: np.ndarray) -> dict:
    qs = np.linspace(0.80, 0.995, 50)
    thresholds = np.quantile(scores, qs)
    best = {"threshold": None, "f1": -1.0, "precision": 0.0, "recall": 0.0}
    for thr in thresholds:
        y_pred = (scores >= thr).astype(int)
        p, r, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="binary", zero_division=0)
        if f1 > best["f1"]:
            best = {"threshold": float(thr), "f1": float(f1), "precision": float(p), "recall": float(r)}
    return best


def _ts_cv_slices(n: int, folds: int) -> list[tuple[slice, slice]]:
    if n < 4 or folds < 1:
        return []
    val_size = max(1, int(n * 0.1))
    min_train = max(val_size, int(n * 0.5))
    if min_train + val_size >= n:
        min_train = max(1, n - val_size - 1)
    step = 0 if folds == 1 else max(1, (n - min_train - val_size) // (folds - 1))
    slices = []
    for i in range(folds):
        train_end = min_train + step * i
        val_start = train_end
        val_end = min(val_start + val_size, n)
        if val_end <= val_start:
            break
        slices.append((slice(0, train_end), slice(val_start, val_end)))
    return slices


def _cv_pr_auc_if(X: np.ndarray, y: np.ndarray, params: dict, folds: int) -> float:
    scores = []
    slices = _ts_cv_slices(len(X), folds)
    for tr, va in slices:
        scaler = StandardScaler()
        X_tr = scaler.fit_transform(X[tr])
        X_va = scaler.transform(X[va])
        model = IsolationForest(**params)
        model.fit(X_tr)
        val_score = -model.decision_function(X_va)
        scores.append(average_precision_score(y[va], val_score))
    return float(np.mean(scores)) if scores else float("nan")


def _cv_pr_auc_prob(model_ctor, X: np.ndarray, y: np.ndarray, folds: int) -> float:
    scores = []
    slices = _ts_cv_slices(len(X), folds)
    for tr, va in slices:
        model = model_ctor()
        sw = _sample_weight(y[tr])
        if sw is None:
            model.fit(X[tr], y[tr])
        else:
            model.fit(X[tr], y[tr], sample_weight=sw)
        val_score = model.predict_proba(X[va])[:, 1]
        scores.append(average_precision_score(y[va], val_score))
    return float(np.mean(scores)) if scores else float("nan")


def main():
    run_dir = make_run_dir("train")

    labeled_path = "data/network_dataset_labeled.csv"
    unlabeled_path = "data/network_dataset.csv"
    has_label = os.path.exists(labeled_path)

    df = load_csv(labeled_path if has_label else unlabeled_path)
    if has_label and "anomaly" not in df.columns:
        raise ValueError("Expected 'anomaly' label column in labeled dataset.")

    df = time_sort(df)
    df_feat = add_time_series_features(df, NUM_COLS)
    feat_cols = get_feature_columns(NUM_COLS)
    base_cols = [c for c in NUM_COLS if c in df_feat.columns]

    if not has_label:
        scaler = StandardScaler()
        X_all = df_feat[feat_cols].astype(float).values
        X_all_s = scaler.fit_transform(X_all)
        model = IsolationForest(**IFOREST_PARAMS)
        model.fit(X_all_s)
        all_scores = -model.decision_function(X_all_s)
        threshold = float(np.quantile(all_scores, ALERT_QUANTILE))
        df_feat = per_row_contribution_z(df_feat, NUM_COLS)
        df_feat["anomaly_score"] = all_scores
        df_feat["is_alert"] = df_feat["anomaly_score"] >= threshold
        df_feat["model_used"] = "IsolationForest"

        joblib.dump(model, run_dir / "model.joblib")
        joblib.dump(scaler, run_dir / "scaler.joblib")
        meta = {
            "num_cols": NUM_COLS,
            "feature_cols": feat_cols,
            "threshold": float(threshold),
            "alert_quantile": ALERT_QUANTILE,
            "model_type": "IsolationForest",
            "has_label": False,
        }
        joblib.dump(meta, run_dir / "meta.joblib")

        df_feat.to_csv(run_dir / "scored_unlabeled.csv", index=False)
        incidents = group_incidents(df_feat, gap_rows=INCIDENT_GAP_ROWS)
        incidents.to_csv(run_dir / "incidents_unlabeled.csv", index=False)

        print("OK Training done.")
        print(f"model=IsolationForest threshold={threshold:.6f}")
        return

    idx_train, idx_val, idx_test = time_split_indices(len(df_feat), TRAIN_RATIO, VAL_RATIO)
    train_df = df_feat.iloc[idx_train].copy()
    val_df = df_feat.iloc[idx_val].copy()
    test_df = df_feat.iloc[idx_test].copy()

    X_train = train_df[feat_cols].astype(float).values
    X_val = val_df[feat_cols].astype(float).values
    X_test = test_df[feat_cols].astype(float).values

    X_train_base = train_df[base_cols].astype(float).values
    X_val_base = val_df[base_cols].astype(float).values
    X_test_base = test_df[base_cols].astype(float).values

    y_train = train_df["anomaly"].astype(int).values
    y_val = val_df["anomaly"].astype(int).values
    y_test = test_df["anomaly"].astype(int).values

    pos_rate = float(y_train.mean())
    contamination = float(np.clip(pos_rate, 0.01, 0.20))
    iforest_params = dict(IFOREST_PARAMS)
    iforest_params["contamination"] = contamination

    cv_summary = {}
    cv_summary["iforest_pr_auc"] = _cv_pr_auc_if(X_train, y_train, iforest_params, CV_FOLDS)
    cv_summary["rf_full_pr_auc"] = _cv_pr_auc_prob(lambda: RandomForestClassifier(**RF_PARAMS), X_train, y_train, CV_FOLDS)
    cv_summary["rf_base_pr_auc"] = _cv_pr_auc_prob(lambda: RandomForestClassifier(**RF_PARAMS), X_train_base, y_train, CV_FOLDS)
    try:
        cv_summary["hgb_pr_auc"] = _cv_pr_auc_prob(
            lambda: HistGradientBoostingClassifier(**HGB_PARAMS), X_train, y_train, CV_FOLDS
        )
    except Exception:
        cv_summary["hgb_pr_auc"] = float("nan")

    candidates = [
        ("IsolationForest", cv_summary["iforest_pr_auc"]),
        ("RandomForest_full", cv_summary["rf_full_pr_auc"]),
        ("RandomForest_base", cv_summary["rf_base_pr_auc"]),
        ("HistGradientBoosting", cv_summary["hgb_pr_auc"]),
    ]
    candidates = [c for c in candidates if not np.isnan(c[1])]
    candidates.sort(key=lambda x: x[1], reverse=True)
    model_choice = candidates[0][0]

    if model_choice == "IsolationForest":
        scaler = StandardScaler()
        X_train_s = scaler.fit_transform(X_train)
        X_val_s = scaler.transform(X_val)
        X_test_s = scaler.transform(X_test)

        model_train = IsolationForest(**iforest_params)
        model_train.fit(X_train_s)
        val_scores = -model_train.decision_function(X_val_s)
        best_val = _tune_threshold(val_scores, y_val)
        threshold = float(best_val["threshold"])
        test_scores = -model_train.decision_function(X_test_s)
        selected_feature_cols = feat_cols
        final_model = model_train
        final_scaler = scaler

    else:
        if model_choice == "RandomForest_base":
            Xtr, Xv, Xt = X_train_base, X_val_base, X_test_base
            selected_feature_cols = base_cols
        else:
            Xtr, Xv, Xt = X_train, X_val, X_test
            selected_feature_cols = feat_cols

        if model_choice == "HistGradientBoosting":
            model_train = HistGradientBoostingClassifier(**HGB_PARAMS)
        else:
            model_train = RandomForestClassifier(**RF_PARAMS)

        sw = _sample_weight(y_train)
        if sw is None:
            model_train.fit(Xtr, y_train)
        else:
            model_train.fit(Xtr, y_train, sample_weight=sw)

        val_scores = model_train.predict_proba(Xv)[:, 1]
        best_val = _tune_threshold(val_scores, y_val)
        threshold = float(best_val["threshold"])
        test_scores = model_train.predict_proba(Xt)[:, 1]
        final_model = model_train
        final_scaler = StandardScaler().fit(Xtr)

    y_pred = (test_scores >= threshold).astype(int)
    p, r, f1, _ = precision_recall_fscore_support(y_test, y_pred, average="binary", zero_division=0)
    roc = roc_auc_score(y_test, test_scores)
    pr = average_precision_score(y_test, test_scores)

    df_feat = per_row_contribution_z(df_feat, NUM_COLS)
    X_all = df_feat[selected_feature_cols].astype(float).values
    if model_choice == "IsolationForest":
        all_scores = -final_model.decision_function(final_scaler.transform(X_all))
    else:
        all_scores = final_model.predict_proba(X_all)[:, 1]

    df_feat["anomaly_score"] = all_scores
    df_feat["is_alert"] = df_feat["anomaly_score"] >= threshold
    df_feat["model_used"] = model_choice

    joblib.dump(final_model, run_dir / "model.joblib")
    joblib.dump(final_scaler, run_dir / "scaler.joblib")

    meta = {
        "num_cols": NUM_COLS,
        "feature_cols": selected_feature_cols,
        "threshold": float(threshold),
        "alert_quantile": ALERT_QUANTILE,
        "iforest_params": iforest_params,
        "rf_params": RF_PARAMS,
        "hgb_params": HGB_PARAMS,
        "incident_gap_rows": INCIDENT_GAP_ROWS,
        "model_type": model_choice,
        "train_ratio": TRAIN_RATIO,
        "val_ratio": VAL_RATIO,
        "has_label": True,
        "feature_cols_iforest": feat_cols,
        "feature_cols_rf_base": base_cols,
        "cv_summary": cv_summary,
        "val_best": best_val,
    }
    metrics = {
        "precision": float(p),
        "recall": float(r),
        "f1": float(f1),
        "roc_auc": float(roc),
        "pr_auc": float(pr),
    }
    joblib.dump(meta, run_dir / "meta.joblib")
    (run_dir / "best_params.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    (run_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    df_feat.to_csv(run_dir / "scored_unlabeled.csv", index=False)
    incidents = group_incidents(df_feat, gap_rows=INCIDENT_GAP_ROWS)
    incidents.to_csv(run_dir / "incidents_unlabeled.csv", index=False)

    print("OK Training done.")
    print(f"model={model_choice} threshold={threshold:.6f}")


if __name__ == "__main__":
    main()
