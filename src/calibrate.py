from __future__ import annotations

"""Probability calibration and threshold selection on labeled data."""

import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import (
    precision_recall_curve,
    average_precision_score,
    precision_recall_fscore_support,
    confusion_matrix,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler

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


def _time_split_idx(n: int, train_ratio: float = 0.70, val_ratio: float = 0.15) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    train_end = int(n * train_ratio)
    val_end = int(n * (train_ratio + val_ratio))
    idx_train = np.arange(0, train_end)
    idx_val = np.arange(train_end, val_end)
    idx_test = np.arange(val_end, n)
    return idx_train, idx_val, idx_test


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
    return {
        "precision": float(p),
        "recall": float(r),
        "f1": float(f1),
        "f2": float(f2),
        "pr_auc": float(pr_auc),
    }


def _plot_pr_curve(run_dir: Path, y_true: np.ndarray, y_score: np.ndarray, name: str) -> None:
    import matplotlib.pyplot as plt

    precision, recall, _ = precision_recall_curve(y_true, y_score)
    plt.figure(figsize=(5, 4))
    plt.plot(recall, precision)
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title(f"PR Curve (Test) - {name}")
    plt.grid(True, alpha=0.3)
    plt.savefig(run_dir / f"pr_curve_{name}.png", dpi=150, bbox_inches="tight")
    plt.close()


def _plot_confusion_matrix(run_dir: Path, y_true: np.ndarray, y_pred: np.ndarray, name: str) -> None:
    import matplotlib.pyplot as plt

    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(4, 4))
    plt.imshow(cm, cmap="Blues")
    plt.title(f"Confusion Matrix (Test) - {name}")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, cm[i, j], ha="center", va="center")
    plt.tight_layout()
    plt.savefig(run_dir / f"confusion_matrix_{name}.png", dpi=150, bbox_inches="tight")
    plt.close()


def _plot_score_hist(run_dir: Path, y_true: np.ndarray, y_score: np.ndarray, name: str) -> None:
    import matplotlib.pyplot as plt

    plt.figure(figsize=(5, 4))
    plt.hist(y_score[y_true == 0], bins=30, alpha=0.7, label="normal")
    plt.hist(y_score[y_true == 1], bins=30, alpha=0.7, label="anomaly")
    plt.title(f"Score Histogram (Test) - {name}")
    plt.xlabel("Score")
    plt.ylabel("Count")
    plt.legend()
    plt.grid(True, alpha=0.2)
    plt.savefig(run_dir / f"score_hist_{name}.png", dpi=150, bbox_inches="tight")
    plt.close()


def _train_and_score(model, X_train, y_train, X_val, X_test) -> dict:
    model.fit(X_train, y_train)
    score_val = model.predict_proba(X_val)[:, 1]
    score_test = model.predict_proba(X_test)[:, 1]
    return {"model": model, "score_val": score_val, "score_test": score_test}


def _fit_sigmoid(scores: np.ndarray, y: np.ndarray) -> LogisticRegression:
    lr = LogisticRegression(max_iter=1000, solver="lbfgs")
    lr.fit(scores.reshape(-1, 1), y)
    return lr


def _apply_sigmoid(model: LogisticRegression, scores: np.ndarray) -> np.ndarray:
    return model.predict_proba(scores.reshape(-1, 1))[:, 1]


def _fit_isotonic(scores: np.ndarray, y: np.ndarray) -> IsotonicRegression:
    iso = IsotonicRegression(out_of_bounds="clip")
    iso.fit(scores, y)
    return iso


def main() -> None:
    run_dir = make_run_dir("calibrate")
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

    idx_train, idx_val, idx_test = _time_split_idx(len(df_feat))
    X_train, X_val, X_test = X[idx_train], X[idx_val], X[idx_test]
    y_train, y_val, y_test = y[idx_train], y[idx_val], y[idx_test]

    models = [
        (
            "LogisticRegression",
            Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="median")),
                    ("scaler", RobustScaler()),
                    ("model", LogisticRegression(max_iter=2000, class_weight="balanced", solver="liblinear")),
                ]
            ),
        ),
        (
            "RandomForest",
            Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="median")),
                    ("model", RandomForestClassifier(n_estimators=400, class_weight="balanced", random_state=42, n_jobs=1)),
                ]
            ),
        ),
        (
            "GradientBoosting",
            Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="median")),
                    ("model", GradientBoostingClassifier(random_state=42)),
                ]
            ),
        ),
    ]

    rows = []
    model_map = {name: model for name, model in models}
    for name, model in models:
        out = _train_and_score(model, X_train, y_train, X_val, X_test)
        best = _best_threshold_by_f2(y_val, out["score_val"])
        test_metrics = _metrics(y_test, out["score_test"], best["threshold"])
        rows.append(
            {
                "model": name,
                "threshold": best["threshold"],
                "val_f2": best["f2"],
                "test_precision": test_metrics["precision"],
                "test_recall": test_metrics["recall"],
                "test_f1": test_metrics["f1"],
                "test_f2": test_metrics["f2"],
                "test_pr_auc": test_metrics["pr_auc"],
            }
        )

    base_best = max(rows, key=lambda r: (r["test_f2"], r["test_recall"]))
    base_name = base_best["model"]
    base_model = model_map[base_name]

    base_model.fit(X_train, y_train)
    base_val = base_model.predict_proba(X_val)[:, 1]
    base_test = base_model.predict_proba(X_test)[:, 1]
    base_thr = _best_threshold_by_f2(y_val, base_val)["threshold"]
    base_metrics = _metrics(y_test, base_test, base_thr)

    calibrations = {}

    sig = _fit_sigmoid(base_val, y_val)
    sig_val = _apply_sigmoid(sig, base_val)
    sig_test = _apply_sigmoid(sig, base_test)
    sig_thr = _best_threshold_by_f2(y_val, sig_val)["threshold"]
    sig_metrics = _metrics(y_test, sig_test, sig_thr)
    calibrations["sigmoid"] = {
        "threshold": sig_thr,
        "metrics": sig_metrics,
        "score_test": sig_test,
    }

    iso = _fit_isotonic(base_val, y_val)
    iso_val = iso.predict(base_val)
    iso_test = iso.predict(base_test)
    iso_thr = _best_threshold_by_f2(y_val, iso_val)["threshold"]
    iso_metrics = _metrics(y_test, iso_test, iso_thr)
    calibrations["isotonic"] = {
        "threshold": iso_thr,
        "metrics": iso_metrics,
        "score_test": iso_test,
    }

    best_cal_name = max(calibrations.keys(), key=lambda k: (calibrations[k]["metrics"]["f2"], calibrations[k]["metrics"]["recall"]))
    best_cal = calibrations[best_cal_name]

    y_pred_base = (base_test >= base_thr).astype(int)
    y_pred_cal = (best_cal["score_test"] >= best_cal["threshold"]).astype(int)

    _plot_pr_curve(run_dir, y_test, base_test, "base")
    _plot_pr_curve(run_dir, y_test, best_cal["score_test"], f"cal_{best_cal_name}")
    _plot_confusion_matrix(run_dir, y_test, y_pred_base, "base")
    _plot_confusion_matrix(run_dir, y_test, y_pred_cal, f"cal_{best_cal_name}")
    _plot_score_hist(run_dir, y_test, base_test, "base")
    _plot_score_hist(run_dir, y_test, best_cal["score_test"], f"cal_{best_cal_name}")

    pred_df = pd.DataFrame(
        {
            "y_true": y_test,
            "score_base": base_test,
            "pred_base": y_pred_base,
            "score_cal": best_cal["score_test"],
            "pred_cal": y_pred_cal,
        }
    )
    pred_df.to_csv(run_dir / "predictions.csv", index=False)

    metrics_payload = {
        "data_path": data_path,
        "feature_cols": feature_cols,
        "kpi_cols": kpi_cols,
        "leakage_cols_removed": leakage_cols,
        "target_like_removed": target_like,
        "base_model": base_name,
        "base_threshold": base_thr,
        "base_metrics_test": base_metrics,
        "best_calibration": best_cal_name,
        "calibrations": {k: {"threshold": v["threshold"], "metrics_test": v["metrics"]} for k, v in calibrations.items()},
        "model_comparison": rows,
    }
    (run_dir / "metrics.json").write_text(json.dumps(metrics_payload, indent=2), encoding="utf-8")

    report_lines = []
    report_lines.append("# Calibration report")
    report_lines.append("")
    report_lines.append(f"- data_path: {data_path}")
    report_lines.append(f"- base_model: {base_name}")
    report_lines.append(f"- base_threshold (val F2): {base_thr:.6f}")
    report_lines.append(f"- base_metrics_test: {base_metrics}")
    report_lines.append(f"- best_calibration: {best_cal_name}")
    report_lines.append(f"- best_cal_metrics_test: {best_cal['metrics']}")
    report_lines.append("")
    report_lines.append("## Model comparison (uncalibrated)")
    report_lines.append("| model | threshold | test_recall | test_f2 | test_f1 | test_pr_auc |")
    report_lines.append("|---|---:|---:|---:|---:|---:|")
    for r in rows:
        report_lines.append(
            f"| {r['model']} | {r['threshold']:.6f} | {r['test_recall']:.4f} | {r['test_f2']:.4f} | {r['test_f1']:.4f} | {r['test_pr_auc']:.4f} |"
        )
    (run_dir / "report.md").write_text("\n".join(report_lines), encoding="utf-8")

    print(f"OK Calibration run saved to {run_dir}")


if __name__ == "__main__":
    main()
