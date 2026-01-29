from __future__ import annotations

import json
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_recall_fscore_support, roc_auc_score, average_precision_score, precision_recall_curve, roc_curve

from .config import NUM_COLS, TRAIN_RATIO, VAL_RATIO
from .data import load_csv
from .eval_utils import time_sort, time_split_indices
from .run_utils import make_run_dir, latest_run_dir


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


def _best_f1_threshold(scores: np.ndarray, y_true: np.ndarray) -> float:
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


def main():
    run_dir = make_run_dir("final")
    tune_dir = latest_run_dir("tune")
    if tune_dir is None:
        raise RuntimeError("No tuning run found. Run: python -m src.tune")

    best = json.loads((tune_dir / "best_params.json").read_text(encoding="utf-8"))
    params = best["params"]

    df = load_csv("data/network_dataset_labeled.csv")
    df = time_sort(df)
    df = df.drop(columns=[c for c in df.columns if c.startswith("anomaly_")], errors="ignore")
    df_feat = _build_features(df, [c for c in NUM_COLS if c in df.columns])

    y = df_feat["anomaly"].astype(int).values
    X = df_feat.drop(columns=["anomaly"], errors="ignore").select_dtypes(include=[np.number]).values

    idx_train, idx_val, idx_test = time_split_indices(len(df_feat), TRAIN_RATIO, VAL_RATIO)
    X_train, X_val, X_test = X[idx_train], X[idx_val], X[idx_test]
    y_train, y_val, y_test = y[idx_train], y[idx_val], y[idx_test]

    X_trainval = np.vstack([X_train, X_val])
    y_trainval = np.hstack([y_train, y_val])
    cv_slices = _ts_cv_slices(len(X_trainval), folds=3)
    thresholds = []
    for tr, va in cv_slices:
        model = RandomForestClassifier(**params)
        model.fit(X_trainval[tr], y_trainval[tr])
        val_scores = model.predict_proba(X_trainval[va])[:, 1]
        thresholds.append(_best_f1_threshold(val_scores, y_trainval[va]))
    threshold = float(np.median(thresholds)) if thresholds else float(np.quantile(y_trainval, 0.99))

    model = RandomForestClassifier(**params)
    model.fit(X_trainval, y_trainval)
    test_scores = model.predict_proba(X_test)[:, 1]
    y_pred = (test_scores >= threshold).astype(int)

    p, r, f1, _ = precision_recall_fscore_support(y_test, y_pred, average="binary", zero_division=0)
    roc = roc_auc_score(y_test, test_scores)
    pr = average_precision_score(y_test, test_scores)

    metrics = {
        "model": "RandomForest",
        "threshold": float(threshold),
        "precision": float(p),
        "recall": float(r),
        "f1": float(f1),
        "roc_auc": float(roc),
        "pr_auc": float(pr),
    }

    all_scores = model.predict_proba(X)[:, 1]
    df_feat["anomaly_score"] = all_scores
    df_feat["is_alert"] = df_feat["anomaly_score"] >= threshold
    df_feat["model_used"] = "RandomForest"

    (run_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    (run_dir / "best_params.json").write_text(json.dumps(best, indent=2), encoding="utf-8")
    df_feat.to_csv(run_dir / "scored_labeled.csv", index=False)

    import matplotlib.pyplot as plt
    prec, rec, _ = precision_recall_curve(y_test, test_scores)
    fpr, tpr, _ = roc_curve(y_test, test_scores)

    plt.figure(figsize=(5, 4))
    plt.plot(rec, prec)
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("PR Curve (Final)")
    plt.grid(True, alpha=0.3)
    plt.savefig(run_dir / "pr_curve.png", dpi=150, bbox_inches="tight")
    plt.close()

    plt.figure(figsize=(5, 4))
    plt.plot(fpr, tpr)
    plt.plot([0, 1], [0, 1], linestyle="--", alpha=0.5)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve (Final)")
    plt.grid(True, alpha=0.3)
    plt.savefig(run_dir / "roc_curve.png", dpi=150, bbox_inches="tight")
    plt.close()

    summary = {
        "why_chosen": "Selected tuned RandomForest based on validation PR-AUC and stable CV threshold.",
        "risk_notes": [
            "Validation to test gap suggests potential distribution shift.",
            "Threshold is CV-based to reduce overfitting to a single split.",
        ],
    }
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"OK Final run saved {run_dir}")
    print(metrics)


if __name__ == "__main__":
    main()
