from __future__ import annotations

"""Rolling time-series cross-validation for leakage-safe evaluation."""

import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    precision_recall_curve,
    average_precision_score,
    precision_recall_fscore_support,
    confusion_matrix,
)
from sklearn.pipeline import Pipeline

from .config import NUM_COLS
from .run_utils import make_run_dir


def _find_data_path() -> str:
    candidates = [
        "/mnt/data/network_dataset_labeled.csv",
        "data/network_dataset_labeled.csv",
    ]
    for p in candidates:
        if os.path.exists(p):
            return p
    raise FileNotFoundError("Could not find network_dataset_labeled.csv in /mnt/data or data/.")


def _time_sort(df: pd.DataFrame) -> pd.DataFrame:
    if "timestamp" not in df.columns:
        raise ValueError("Expected timestamp column for chronological split.")
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce", utc=True)
    df = df.dropna(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)
    return df


def _build_features(df: pd.DataFrame, kpi_cols: list[str]) -> pd.DataFrame:
    out = df.copy()
    for c in kpi_cols:
        out[c] = pd.to_numeric(out[c], errors="coerce")
    for c in kpi_cols:
        s = out[c]
        s1 = s.shift(1)
        out[f"{c}__lag1"] = s1
        out[f"{c}__lag2"] = s.shift(2)
        out[f"{c}__lag3"] = s.shift(3)
        out[f"{c}__delta1"] = s - s1
        for w in (3, 6):
            out[f"{c}__roll{w}_mean"] = s1.rolling(w, min_periods=1).mean()
            out[f"{c}__roll{w}_std"] = s1.rolling(w, min_periods=1).std()
    return out


def _fbeta(precision: np.ndarray, recall: np.ndarray, beta: float = 2.0) -> np.ndarray:
    b2 = beta * beta
    denom = (b2 * precision + recall)
    with np.errstate(divide="ignore", invalid="ignore"):
        f = (1 + b2) * precision * recall / denom
    f[np.isnan(f)] = 0.0
    return f


def _best_threshold_by_f2(y_true: np.ndarray, y_score: np.ndarray) -> dict:
    precision, recall, thresholds = precision_recall_curve(y_true, y_score)
    f2 = _fbeta(precision, recall, beta=2.0)
    if len(thresholds) == 0:
        return {"threshold": 0.5, "precision": 0.0, "recall": 0.0, "f2": 0.0}
    f2_use = f2[:-1]
    best_idx = int(np.argmax(f2_use))
    return {
        "threshold": float(thresholds[best_idx]),
        "precision": float(precision[best_idx]),
        "recall": float(recall[best_idx]),
        "f2": float(f2_use[best_idx]),
    }


def _metrics(y_true: np.ndarray, y_score: np.ndarray, threshold: float) -> dict:
    y_pred = (y_score >= threshold).astype(int)
    p, r, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="binary", zero_division=0)
    pr_auc = average_precision_score(y_true, y_score)
    f2 = _fbeta(np.array([p]), np.array([r]), beta=2.0)[0]
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    return {
        "precision": float(p),
        "recall": float(r),
        "f1": float(f1),
        "f2": float(f2),
        "pr_auc": float(pr_auc),
        "tp": int(tp),
        "fp": int(fp),
        "fn": int(fn),
        "tn": int(tn),
    }


def _plot_pr_curve(run_dir: Path, y_true: np.ndarray, y_score: np.ndarray, fold_id: int) -> None:
    import matplotlib.pyplot as plt

    precision, recall, _ = precision_recall_curve(y_true, y_score)
    plt.figure(figsize=(5, 4))
    plt.plot(recall, precision)
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title(f"PR Curve (Fold {fold_id})")
    plt.grid(True, alpha=0.3)
    plt.savefig(run_dir / f"pr_curve_fold{fold_id}.png", dpi=150, bbox_inches="tight")
    plt.close()


def _plot_confusion_matrix(run_dir: Path, y_true: np.ndarray, y_pred: np.ndarray, fold_id: int) -> None:
    import matplotlib.pyplot as plt

    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(4, 4))
    plt.imshow(cm, cmap="Blues")
    plt.title(f"Confusion Matrix (Fold {fold_id})")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, cm[i, j], ha="center", va="center")
    plt.tight_layout()
    plt.savefig(run_dir / f"confusion_matrix_fold{fold_id}.png", dpi=150, bbox_inches="tight")
    plt.close()


def _plot_score_hist(run_dir: Path, y_true: np.ndarray, y_score: np.ndarray, fold_id: int) -> None:
    import matplotlib.pyplot as plt

    plt.figure(figsize=(5, 4))
    plt.hist(y_score[y_true == 0], bins=30, alpha=0.7, label="normal")
    plt.hist(y_score[y_true == 1], bins=30, alpha=0.7, label="anomaly")
    plt.title(f"Score Histogram (Fold {fold_id})")
    plt.xlabel("Score")
    plt.ylabel("Count")
    plt.legend()
    plt.grid(True, alpha=0.2)
    plt.savefig(run_dir / f"score_hist_fold{fold_id}.png", dpi=150, bbox_inches="tight")
    plt.close()


def _rolling_splits(n: int, train_frac: float = 0.60, val_frac: float = 0.15, test_frac: float = 0.15, step_frac: float = 0.10) -> list[tuple[np.ndarray, np.ndarray, np.ndarray]]:
    splits = []
    train_size = int(n * train_frac)
    val_size = int(n * val_frac)
    test_size = int(n * test_frac)
    step = max(1, int(n * step_frac))

    start = 0
    while True:
        train_end = start + train_size
        val_end = train_end + val_size
        test_end = val_end + test_size
        if test_end > n:
            break
        idx_train = np.arange(0, train_end)
        idx_val = np.arange(train_end, val_end)
        idx_test = np.arange(val_end, test_end)
        splits.append((idx_train, idx_val, idx_test))
        start += step
    return splits


def main() -> None:
    run_dir = make_run_dir("timecv")
    data_path = _find_data_path()
    df = pd.read_csv(data_path)
    df = _time_sort(df)

    leakage_cols = [c for c in df.columns if c.startswith("anomaly_")]
    target_like = [c for c in df.columns if "target" in c.lower()]
    drop_cols = set(leakage_cols + target_like + ["anomaly", "timestamp"])

    kpi_cols = [c for c in NUM_COLS if c in df.columns]
    df_feat = _build_features(df, kpi_cols)

    numeric_cols = df_feat.select_dtypes(include=[np.number]).columns.tolist()
    feature_cols = [c for c in numeric_cols if c not in drop_cols]

    X = df_feat[feature_cols].values
    y = df_feat["anomaly"].astype(int).values

    splits = _rolling_splits(len(df_feat))
    if not splits:
        raise RuntimeError("Not enough data to create rolling splits.")

    model = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("model", GradientBoostingClassifier(random_state=42)),
        ]
    )

    rows = []
    for i, (idx_train, idx_val, idx_test) in enumerate(splits, start=1):
        X_train, X_val, X_test = X[idx_train], X[idx_val], X[idx_test]
        y_train, y_val, y_test = y[idx_train], y[idx_val], y[idx_test]

        model.fit(X_train, y_train)
        score_val = model.predict_proba(X_val)[:, 1]
        score_test = model.predict_proba(X_test)[:, 1]

        best = _best_threshold_by_f2(y_val, score_val)
        metrics = _metrics(y_test, score_test, best["threshold"])

        y_pred = (score_test >= best["threshold"]).astype(int)

        _plot_pr_curve(run_dir, y_test, score_test, i)
        _plot_confusion_matrix(run_dir, y_test, y_pred, i)
        _plot_score_hist(run_dir, y_test, score_test, i)

        rows.append(
            {
                "fold": i,
                "train_start": int(idx_train[0]),
                "train_end": int(idx_train[-1]),
                "val_start": int(idx_val[0]),
                "val_end": int(idx_val[-1]),
                "test_start": int(idx_test[0]),
                "test_end": int(idx_test[-1]),
                "threshold": best["threshold"],
                **metrics,
            }
        )

    metrics_df = pd.DataFrame(rows)
    metrics_df.to_csv(run_dir / "metrics_per_fold.csv", index=False)

    summary = {}
    for key in ["precision", "recall", "f1", "f2", "pr_auc"]:
        summary[key] = {
            "mean": float(metrics_df[key].mean()),
            "std": float(metrics_df[key].std(ddof=0)),
        }
    summary["folds"] = int(len(rows))

    (run_dir / "metrics_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    report_lines = []
    report_lines.append("# Rolling time-CV report")
    report_lines.append("")
    report_lines.append(f"- data_path: {data_path}")
    report_lines.append(f"- feature_set: A (no anomaly_* and no target-like columns)")
    report_lines.append(f"- model: GradientBoostingClassifier")
    report_lines.append(f"- folds: {len(rows)}")
    report_lines.append("")
    report_lines.append("## Summary (mean ± std)")
    for key in ["precision", "recall", "f1", "f2", "pr_auc"]:
        report_lines.append(f"- {key}: {summary[key]['mean']:.4f} ± {summary[key]['std']:.4f}")
    report_lines.append("")
    report_lines.append("See metrics_per_fold.csv for per-fold details and plots per fold.")
    (run_dir / "report.md").write_text("\n".join(report_lines), encoding="utf-8")

    print(f"OK TimeCV run saved to {run_dir}")


if __name__ == "__main__":
    main()
