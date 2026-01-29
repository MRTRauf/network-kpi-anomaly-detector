from __future__ import annotations

"""Rolling time-series CV for multi-model benchmarking with F2 thresholding."""

import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    precision_recall_curve,
    precision_recall_fscore_support,
)
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import Pipeline

from .config import NUM_COLS
from .run_utils import make_run_dir


def _find_data_path() -> str:
    candidates = [
        "/mnt/data/network_dataset_labeled.csv",
        "data/network_dataset_labeled.csv",
    ]
    for path in candidates:
        if os.path.exists(path):
            return path
    raise FileNotFoundError("Could not find network_dataset_labeled.csv in /mnt/data or data/.")


def _time_sort(df: pd.DataFrame) -> pd.DataFrame:
    if "timestamp" not in df.columns:
        raise ValueError("Expected timestamp column for chronological split.")
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce", utc=True)
    df = df.dropna(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)
    return df


def _build_features(df: pd.DataFrame, kpi_cols: list[str]) -> pd.DataFrame:
    out = df.copy()
    for col in kpi_cols:
        out[col] = pd.to_numeric(out[col], errors="coerce")
    for col in kpi_cols:
        s = out[col]
        s1 = s.shift(1)
        out[f"{col}__lag1"] = s1
        out[f"{col}__lag2"] = s.shift(2)
        out[f"{col}__lag3"] = s.shift(3)
        out[f"{col}__delta1"] = s - s1
        for w in (3, 6):
            out[f"{col}__roll{w}_mean"] = s1.rolling(w, min_periods=1).mean()
            out[f"{col}__roll{w}_std"] = s1.rolling(w, min_periods=1).std()
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


def _rolling_splits_with_min_anoms(
    y: np.ndarray,
    n: int,
    desired_folds: int = 5,
    min_test_anoms: int = 5,
    train_frac: float = 0.60,
    val_frac: float = 0.15,
    test_frac: float = 0.15,
) -> list[tuple[np.ndarray, np.ndarray, np.ndarray]]:
    train_size = int(n * train_frac)
    val_size = int(n * val_frac)
    test_size = int(n * test_frac)

    for folds in range(desired_folds, 0, -1):
        step = max(1, int((n - (train_size + val_size + test_size)) / max(folds - 1, 1)))
        splits: list[tuple[np.ndarray, np.ndarray, np.ndarray]] = []
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
            if y[idx_test].sum() >= min_test_anoms:
                splits.append((idx_train, idx_val, idx_test))
            start += step
        if splits:
            return splits[:folds]
    return []


def _plot_pr_curve(run_dir: Path, y_true: np.ndarray, y_score: np.ndarray, tag: str) -> None:
    import matplotlib.pyplot as plt

    precision, recall, _ = precision_recall_curve(y_true, y_score)
    plt.figure(figsize=(5, 4))
    plt.plot(recall, precision)
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title(f"PR Curve ({tag})")
    plt.grid(True, alpha=0.3)
    plt.savefig(run_dir / f"pr_curve_{tag}.png", dpi=150, bbox_inches="tight")
    plt.close()


def _plot_confusion_matrix(run_dir: Path, y_true: np.ndarray, y_pred: np.ndarray, tag: str) -> None:
    import matplotlib.pyplot as plt

    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(4, 4))
    plt.imshow(cm, cmap="Blues")
    plt.title(f"Confusion Matrix ({tag})")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, cm[i, j], ha="center", va="center")
    plt.tight_layout()
    plt.savefig(run_dir / f"confusion_matrix_{tag}.png", dpi=150, bbox_inches="tight")
    plt.close()


def main() -> None:
    run_dir = make_run_dir("timecv_models")
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

    splits = _rolling_splits_with_min_anoms(y, len(df_feat), desired_folds=5, min_test_anoms=5)
    if not splits:
        raise RuntimeError("Not enough data/anomalies to create rolling splits with min_test_anoms=5.")

    models = {
        "LogisticRegression": Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("model", LogisticRegression(max_iter=2000, class_weight="balanced", solver="liblinear")),
            ]
        ),
        "RandomForest": Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("model", RandomForestClassifier(n_estimators=400, class_weight="balanced", random_state=42, n_jobs=1)),
            ]
        ),
        "GradientBoosting": Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("model", GradientBoostingClassifier(random_state=42)),
            ]
        ),
    }

    rows = []
    for model_name, model in models.items():
        for fold_idx, (idx_train, idx_val, idx_test) in enumerate(splits, start=1):
            X_train, X_val, X_test = X[idx_train], X[idx_val], X[idx_test]
            y_train, y_val, y_test = y[idx_train], y[idx_val], y[idx_test]

            model.fit(X_train, y_train)
            score_val = model.predict_proba(X_val)[:, 1]
            score_test = model.predict_proba(X_test)[:, 1]

            best = _best_threshold_by_f2(y_val, score_val)
            metrics = _metrics(y_test, score_test, best["threshold"])

            y_pred = (score_test >= best["threshold"]).astype(int)
            tag = f"{model_name}_fold{fold_idx}"
            _plot_pr_curve(run_dir, y_test, score_test, tag)
            _plot_confusion_matrix(run_dir, y_test, y_pred, tag)

            rows.append(
                {
                    "model": model_name,
                    "fold": fold_idx,
                    "threshold": best["threshold"],
                    **metrics,
                }
            )

    metrics_df = pd.DataFrame(rows)
    metrics_df.to_csv(run_dir / "metrics_per_fold.csv", index=False)

    summary: dict[str, dict[str, dict[str, float]]] = {}
    for model_name in models.keys():
        subset = metrics_df[metrics_df["model"] == model_name]
        summary[model_name] = {}
        for key in ["precision", "recall", "f1", "f2", "pr_auc"]:
            summary[model_name][key] = {
                "mean": float(subset[key].mean()),
                "std": float(subset[key].std(ddof=0)),
            }

    best_model = max(
        summary.keys(),
        key=lambda name: (summary[name]["f2"]["mean"], summary[name]["recall"]["mean"]),
    )

    calibration = {}
    if best_model in models:
        calibrations = []
        for fold_idx, (idx_train, idx_val, idx_test) in enumerate(splits, start=1):
            X_train, X_val, X_test = X[idx_train], X[idx_val], X[idx_test]
            y_train, y_val, y_test = y[idx_train], y[idx_val], y[idx_test]

            base = models[best_model]
            tscv = TimeSeriesSplit(n_splits=3)
            cal = CalibratedClassifierCV(base, method="sigmoid", cv=tscv)
            cal.fit(X_train, y_train)
            score_val = cal.predict_proba(X_val)[:, 1]
            score_test = cal.predict_proba(X_test)[:, 1]

            best = _best_threshold_by_f2(y_val, score_val)
            metrics = _metrics(y_test, score_test, best["threshold"])
            calibrations.append(
                {
                    "fold": fold_idx,
                    "threshold": best["threshold"],
                    **metrics,
                }
            )
        cal_df = pd.DataFrame(calibrations)
        calibration["sigmoid"] = {
            "per_fold": calibrations,
            "summary": {
                key: {
                    "mean": float(cal_df[key].mean()),
                    "std": float(cal_df[key].std(ddof=0)),
                }
                for key in ["precision", "recall", "f1", "f2", "pr_auc"]
            },
        }

    metrics_summary = {
        "feature_set": "A",
        "min_test_anoms": 5,
        "folds": int(len(splits)),
        "models": summary,
        "best_model": best_model,
        "calibration": calibration,
    }
    (run_dir / "metrics_summary.json").write_text(json.dumps(metrics_summary, indent=2), encoding="utf-8")

    report_lines = []
    report_lines.append("# Time-CV model comparison")
    report_lines.append("")
    report_lines.append(f"- data_path: {data_path}")
    report_lines.append("- leakage controls: drop anomaly_* and target-like columns")
    report_lines.append(f"- folds_used: {len(splits)}")
    report_lines.append(f"- best_model: {best_model}")
    report_lines.append("")
    report_lines.append("## Summary (mean ± std)")
    report_lines.append("| model | precision | recall | f1 | f2 | pr_auc |")
    report_lines.append("|---|---:|---:|---:|---:|---:|")
    for name in models.keys():
        s = summary[name]
        report_lines.append(
            f"| {name} | {s['precision']['mean']:.4f} ± {s['precision']['std']:.4f} | "
            f"{s['recall']['mean']:.4f} ± {s['recall']['std']:.4f} | "
            f"{s['f1']['mean']:.4f} ± {s['f1']['std']:.4f} | "
            f"{s['f2']['mean']:.4f} ± {s['f2']['std']:.4f} | "
            f"{s['pr_auc']['mean']:.4f} ± {s['pr_auc']['std']:.4f} |"
        )
    (run_dir / "report_snippet.md").write_text("\n".join(report_lines), encoding="utf-8")

    print(f"OK TimeCV models run saved to {run_dir}")


if __name__ == "__main__":
    main()
