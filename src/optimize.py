from __future__ import annotations

"""Model comparison and F2-optimized thresholding on labeled data."""

import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    precision_recall_curve,
    average_precision_score,
    precision_recall_fscore_support,
    confusion_matrix,
)
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier, GradientBoostingClassifier
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


def _audit(df: pd.DataFrame) -> dict:
    out = {
        "rows": int(len(df)),
        "cols": int(len(df.columns)),
        "anomaly_rate": float(df["anomaly"].mean()) if "anomaly" in df.columns else None,
        "missing_top": df.isna().mean().sort_values(ascending=False).head(10).to_dict(),
        "object_cols": df.select_dtypes(include=["object"]).columns.tolist(),
    }
    return out


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


def _operating_point_recall(y_true: np.ndarray, y_score: np.ndarray, min_recall: float = 0.70) -> dict | None:
    precision, recall, thresholds = precision_recall_curve(y_true, y_score)
    if len(thresholds) == 0:
        return None
    f2 = _fbeta(precision, recall, beta=2.0)[:-1]
    candidates = []
    for i, thr in enumerate(thresholds):
        if recall[i] >= min_recall:
            candidates.append((f2[i], precision[i], recall[i], thr))
    if not candidates:
        return None
    best = max(candidates, key=lambda x: x[0])
    return {
        "threshold": float(best[3]),
        "precision": float(best[1]),
        "recall": float(best[2]),
        "f2": float(best[0]),
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


def _plot_pr_curve(run_dir: Path, y_true: np.ndarray, y_score: np.ndarray) -> None:
    import matplotlib.pyplot as plt

    precision, recall, _ = precision_recall_curve(y_true, y_score)
    plt.figure(figsize=(5, 4))
    plt.plot(recall, precision)
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("PR Curve (Test)")
    plt.grid(True, alpha=0.3)
    plt.savefig(run_dir / "pr_curve.png", dpi=150, bbox_inches="tight")
    plt.close()


def _plot_confusion_matrix(run_dir: Path, y_true: np.ndarray, y_pred: np.ndarray) -> None:
    import matplotlib.pyplot as plt

    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(4, 4))
    plt.imshow(cm, cmap="Blues")
    plt.title("Confusion Matrix (Test)")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, cm[i, j], ha="center", va="center")
    plt.tight_layout()
    plt.savefig(run_dir / "confusion_matrix.png", dpi=150, bbox_inches="tight")
    plt.close()


def _plot_score_hist(run_dir: Path, y_true: np.ndarray, y_score: np.ndarray) -> None:
    import matplotlib.pyplot as plt

    plt.figure(figsize=(5, 4))
    plt.hist(y_score[y_true == 0], bins=30, alpha=0.7, label="normal")
    plt.hist(y_score[y_true == 1], bins=30, alpha=0.7, label="anomaly")
    plt.title("Score Histogram (Test)")
    plt.xlabel("Score")
    plt.ylabel("Count")
    plt.legend()
    plt.grid(True, alpha=0.2)
    plt.savefig(run_dir / "score_hist.png", dpi=150, bbox_inches="tight")
    plt.close()


def _train_and_score(model_name: str, model, X_train, y_train, X_val, X_test) -> dict:
    model.fit(X_train, y_train)
    score_val = model.predict_proba(X_val)[:, 1]
    score_test = model.predict_proba(X_test)[:, 1]
    return {"model": model, "score_val": score_val, "score_test": score_test, "model_name": model_name}


def main() -> None:
    run_dir = make_run_dir("optimize")
    data_path = _find_data_path()
    df = pd.read_csv(data_path)

    audit = _audit(df)
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

    results = []
    models = []

    lr_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", RobustScaler()),
            ("model", LogisticRegression(max_iter=2000, class_weight="balanced", solver="liblinear")),
        ]
    )
    models.append(("LogisticRegression", lr_pipe))

    rf_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("model", RandomForestClassifier(n_estimators=400, class_weight="balanced", random_state=42, n_jobs=1)),
        ]
    )
    models.append(("RandomForest", rf_pipe))

    hgb_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("model", HistGradientBoostingClassifier(max_depth=5, max_iter=300, learning_rate=0.1, random_state=42)),
        ]
    )
    models.append(("HistGradientBoosting", hgb_pipe))

    model_map = {name: model for name, model in models}

    for name, model in models:
        try:
            out = _train_and_score(name, model, X_train, y_train, X_val, X_test)
        except Exception as exc:
            results.append(
                {
                    "model": name,
                    "status": "failed",
                    "error": str(exc),
                }
            )
            if name == "HistGradientBoosting":
                fallback = Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="median")),
                        ("model", GradientBoostingClassifier(random_state=42)),
                    ]
                )
                try:
                    out_fb = _train_and_score("GradientBoosting_fallback", fallback, X_train, y_train, X_val, X_test)
                except Exception as exc_fb:
                    results.append(
                        {
                            "model": "GradientBoosting_fallback",
                            "status": "failed",
                            "error": str(exc_fb),
                        }
                    )
                else:
                    best_fb = _best_threshold_by_f2(y_val, out_fb["score_val"])
                    op70_fb = _operating_point_recall(y_val, out_fb["score_val"], min_recall=0.70)
                    test_metrics_fb = _metrics(y_test, out_fb["score_test"], best_fb["threshold"])
                    results.append(
                        {
                            "model": "GradientBoosting_fallback",
                            "status": "ok",
                            "threshold": best_fb["threshold"],
                            "val_f2": best_fb["f2"],
                            "test_precision": test_metrics_fb["precision"],
                            "test_recall": test_metrics_fb["recall"],
                            "test_f1": test_metrics_fb["f1"],
                            "test_f2": test_metrics_fb["f2"],
                            "test_pr_auc": test_metrics_fb["pr_auc"],
                            "op_recall70": op70_fb,
                        }
                    )
                    model_map["GradientBoosting_fallback"] = fallback
            continue

        best = _best_threshold_by_f2(y_val, out["score_val"])
        op70 = _operating_point_recall(y_val, out["score_val"], min_recall=0.70)
        test_metrics = _metrics(y_test, out["score_test"], best["threshold"])

        results.append(
            {
                "model": name,
                "status": "ok",
                "threshold": best["threshold"],
                "val_f2": best["f2"],
                "test_precision": test_metrics["precision"],
                "test_recall": test_metrics["recall"],
                "test_f1": test_metrics["f1"],
                "test_f2": test_metrics["f2"],
                "test_pr_auc": test_metrics["pr_auc"],
                "op_recall70": op70,
            }
        )

    results_ok = [r for r in results if r.get("status") == "ok"]
    if not results_ok:
        raise RuntimeError("All models failed; see results for errors.")

    best_row = max(results_ok, key=lambda r: (r["test_f2"], r["test_recall"]))
    best_name = best_row["model"]

    best_model = model_map.get(best_name)
    if best_model is None:
        raise RuntimeError("Best model not found in candidates.")

    best_model.fit(X_train, y_train)
    score_val = best_model.predict_proba(X_val)[:, 1]
    score_test = best_model.predict_proba(X_test)[:, 1]
    best_thr = _best_threshold_by_f2(y_val, score_val)["threshold"]
    op70 = _operating_point_recall(y_val, score_val, min_recall=0.70)

    y_pred_test = (score_test >= best_thr).astype(int)
    best_metrics = _metrics(y_test, score_test, best_thr)

    rng = np.random.RandomState(42)
    y_test_shuf = rng.permutation(y_test)
    pr_auc_shuf_label = average_precision_score(y_test_shuf, score_test)

    X_test_shuf = X_test.copy()
    block = 50
    for i in range(0, len(X_test_shuf), block):
        idx = np.arange(i, min(i + block, len(X_test_shuf)))
        rng.shuffle(idx)
        X_test_shuf[i : i + len(idx)] = X_test_shuf[idx]
    score_test_shuf = best_model.predict_proba(X_test_shuf)[:, 1]
    pr_auc_shuf_rows = average_precision_score(y_test, score_test_shuf)

    metrics_payload = {
        "data_path": data_path,
        "feature_cols": feature_cols,
        "kpi_cols": kpi_cols,
        "leakage_cols_removed": leakage_cols,
        "target_like_removed": target_like,
        "best_model": best_name,
        "threshold": best_thr,
        "metrics_test": best_metrics,
        "operating_point_recall_0_70": op70,
        "sanity": {
            "pr_auc_shuffled_labels": float(pr_auc_shuf_label),
            "pr_auc_shuffled_rows": float(pr_auc_shuf_rows),
        },
        "all_models": results,
    }
    (run_dir / "metrics.json").write_text(json.dumps(metrics_payload, indent=2), encoding="utf-8")

    pred_df = pd.DataFrame(
        {
            "y_true": y_test,
            "y_score": score_test,
            "y_pred": y_pred_test,
        }
    )
    pred_df.to_csv(run_dir / "predictions.csv", index=False)

    _plot_pr_curve(run_dir, y_test, score_test)
    _plot_confusion_matrix(run_dir, y_test, y_pred_test)
    _plot_score_hist(run_dir, y_test, score_test)

    report_lines = []
    report_lines.append("# Optimization report")
    report_lines.append("")
    report_lines.append("## Data audit")
    report_lines.append(f"- data_path: {data_path}")
    report_lines.append(f"- rows: {audit['rows']}")
    report_lines.append(f"- cols: {audit['cols']}")
    report_lines.append(f"- anomaly_rate: {audit['anomaly_rate']:.4f}")
    report_lines.append(f"- object_cols: {audit['object_cols']}")
    report_lines.append(f"- leakage_cols_removed: {leakage_cols}")
    report_lines.append(f"- target_like_removed: {target_like}")
    report_lines.append("")
    report_lines.append("## Models tried")
    report_lines.append("| model | status | threshold | test_recall | test_f2 | test_f1 | test_pr_auc |")
    report_lines.append("|---|---|---:|---:|---:|---:|---:|")
    for r in results:
        if r.get("status") != "ok":
            report_lines.append(f"| {r['model']} | failed | - | - | - | - | - |")
            continue
        report_lines.append(
            f"| {r['model']} | ok | {r['threshold']:.4f} | {r['test_recall']:.4f} | {r['test_f2']:.4f} | {r['test_f1']:.4f} | {r['test_pr_auc']:.4f} |"
        )
    report_lines.append("")
    report_lines.append("## Best model")
    report_lines.append(f"- best_model: {best_name}")
    report_lines.append(f"- threshold (val max F2): {best_thr:.4f}")
    report_lines.append(f"- test metrics: {best_metrics}")
    if op70 is not None:
        report_lines.append(f"- operating_point_recall>=0.70: {op70}")
    else:
        report_lines.append("- operating_point_recall>=0.70: not achievable on validation")
    report_lines.append("")
    report_lines.append("## Sanity checks")
    report_lines.append(f"- pr_auc_shuffled_labels: {pr_auc_shuf_label:.4f}")
    report_lines.append(f"- pr_auc_shuffled_rows: {pr_auc_shuf_rows:.4f}")
    report_lines.append("")
    report_lines.append("## Feature sets")
    report_lines.append("- A: excludes anomaly_* and target-like columns (used).")
    report_lines.append("- B: anomaly_* not included due to leakage risk (skipped).")

    (run_dir / "report.md").write_text("\n".join(report_lines), encoding="utf-8")

    print(f"OK Optimization run saved to {run_dir}")


if __name__ == "__main__":
    main()
