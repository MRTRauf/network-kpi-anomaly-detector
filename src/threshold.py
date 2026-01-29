from __future__ import annotations

import json
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_recall_fscore_support, precision_recall_curve

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


def _best_f1_threshold(scores: np.ndarray, y_true: np.ndarray) -> dict:
    qs = np.linspace(0.80, 0.995, 50)
    thresholds = np.quantile(scores, qs)
    best = {"threshold": float(thresholds[0]), "f1": -1.0, "precision": 0.0, "recall": 0.0}
    for thr in thresholds:
        y_pred = (scores >= thr).astype(int)
        p, r, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="binary", zero_division=0)
        if f1 > best["f1"]:
            best = {"threshold": float(thr), "f1": float(f1), "precision": float(p), "recall": float(r)}
    return best


def _threshold_for_precision(scores: np.ndarray, y_true: np.ndarray, target_p: float) -> dict:
    prec, rec, thr = precision_recall_curve(y_true, scores)
    best = None
    for p, r, t in zip(prec, rec, np.append(thr, thr[-1] if len(thr) else 1.0)):
        if p >= target_p:
            if best is None or r > best["recall"]:
                best = {"threshold": float(t), "precision": float(p), "recall": float(r)}
    return best or {"threshold": float(np.quantile(scores, 0.99)), "precision": 0.0, "recall": 0.0}


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


def main():
    run_dir = make_run_dir("threshold")
    tune_dir = latest_run_dir("tune")
    if tune_dir is None:
        raise RuntimeError("No tuning run found. Run: python -m src.tune")

    best_params = json.loads((tune_dir / "best_params.json").read_text(encoding="utf-8"))
    params = best_params["params"]

    df = load_csv("data/network_dataset_labeled.csv")
    if "anomaly" not in df.columns:
        raise ValueError("Expected 'anomaly' label column in labeled dataset.")

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

    f1_thresholds = []
    hp_thresholds = []

    for tr, va in cv_slices:
        model = RandomForestClassifier(**params)
        model.fit(X_trainval[tr], y_trainval[tr])
        val_scores = model.predict_proba(X_trainval[va])[:, 1]
        best_f1 = _best_f1_threshold(val_scores, y_trainval[va])
        high_precision = _threshold_for_precision(val_scores, y_trainval[va], target_p=0.8)
        f1_thresholds.append(best_f1["threshold"])
        hp_thresholds.append(high_precision["threshold"])

    robust_f1_thr = float(np.median(f1_thresholds)) if f1_thresholds else float(np.quantile(y_trainval, 0.99))
    robust_hp_thr = float(np.median(hp_thresholds)) if hp_thresholds else float(np.quantile(y_trainval, 0.99))

    model = RandomForestClassifier(**params)
    model.fit(X_trainval, y_trainval)
    test_scores = model.predict_proba(X_test)[:, 1]

    y_pred_f1 = (test_scores >= robust_f1_thr).astype(int)
    p, r, f1, _ = precision_recall_fscore_support(y_test, y_pred_f1, average="binary", zero_division=0)

    y_pred_hp = (test_scores >= robust_hp_thr).astype(int)
    p2, r2, f12, _ = precision_recall_fscore_support(y_test, y_pred_hp, average="binary", zero_division=0)

    out = {
        "model": "RandomForest",
        "threshold_best_f1_cv": float(robust_f1_thr),
        "threshold_high_precision_cv": float(robust_hp_thr),
        "test_metrics_best_f1_cv": {"precision": float(p), "recall": float(r), "f1": float(f1)},
        "test_metrics_high_precision_cv": {"precision": float(p2), "recall": float(r2), "f1": float(f12)},
    }

    (run_dir / "thresholds.json").write_text(json.dumps(out, indent=2), encoding="utf-8")

    scored = df_feat.iloc[idx_test].copy()
    scored["anomaly_score"] = test_scores
    scored["is_alert"] = test_scores >= robust_f1_thr
    scored.to_csv(run_dir / "scored_labeled.csv", index=False)

    print(f"OK Thresholding done. Saved {run_dir}")
    print(out)


if __name__ == "__main__":
    main()
