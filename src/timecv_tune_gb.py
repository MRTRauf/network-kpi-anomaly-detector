from __future__ import annotations

"""Rolling time-series CV tuning for GradientBoosting with operating points."""

import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    precision_recall_curve,
    precision_recall_fscore_support,
)
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
        "alert_rate": float(y_pred.mean()),
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


def _candidate_grid() -> list[dict]:
    candidates = []
    for n_estimators in [100, 200, 300]:
        for learning_rate in [0.03, 0.05, 0.1]:
            for max_depth in [2, 3]:
                for min_samples_leaf in [1, 3, 5]:
                    for subsample in [0.7, 0.85, 1.0]:
                        candidates.append(
                            {
                                "n_estimators": n_estimators,
                                "learning_rate": learning_rate,
                                "max_depth": max_depth,
                                "min_samples_leaf": min_samples_leaf,
                                "subsample": subsample,
                            }
                        )
    return candidates


def _threshold_high_recall(y_true: np.ndarray, y_score: np.ndarray, min_recall: float = 0.90) -> float:
    precision, recall, thresholds = precision_recall_curve(y_true, y_score)
    best_idx = None
    best_precision = -1.0
    for i, thr in enumerate(thresholds):
        if recall[i] >= min_recall and precision[i] > best_precision:
            best_precision = precision[i]
            best_idx = i
    if best_idx is not None:
        return float(thresholds[best_idx])
    return float(thresholds[np.argmax(recall[:-1])]) if len(thresholds) else 0.5


def _threshold_alert_budget(y_score: np.ndarray, target_rate: float = 0.05) -> float:
    q = 1.0 - target_rate
    return float(np.quantile(y_score, q))


def _top_feature_explanations(row: pd.Series, kpi_cols: list[str]) -> list[str]:
    scores = []
    for kpi in kpi_cols:
        val = row.get(kpi, np.nan)
        roll = row.get(f"{kpi}__roll3_mean", np.nan)
        delta = row.get(f"{kpi}__delta1", np.nan)
        if pd.isna(val) or pd.isna(roll) or pd.isna(delta):
            continue
        score = abs(val - roll) + abs(delta)
        scores.append((score, kpi, val, roll, delta))
    scores.sort(reverse=True, key=lambda x: x[0])
    top = []
    for _, kpi, val, roll, delta in scores[:3]:
        top.append(f"{kpi}: value={val:.3f}, roll3_mean={roll:.3f}, delta1={delta:.3f}")
    return top


def main() -> None:
    run_dir = make_run_dir("timecv_tune_gb")
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

    candidates = _candidate_grid()
    candidate_rows = []
    evaluated = 0
    best_f2 = -1.0
    checkpoint_f2 = None
    checkpoint_count = 0
    early_stop_reason = None

    for params in candidates:
        if evaluated >= 60:
            early_stop_reason = "max_configs_reached"
            break
        fold_metrics = []
        for idx_train, idx_val, idx_test in splits:
            model = Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="median")),
                    (
                        "model",
                        GradientBoostingClassifier(
                            random_state=42,
                            n_estimators=params["n_estimators"],
                            learning_rate=params["learning_rate"],
                            max_depth=params["max_depth"],
                            min_samples_leaf=params["min_samples_leaf"],
                            subsample=params["subsample"],
                        ),
                    ),
                ]
            )

            X_train, X_val, X_test = X[idx_train], X[idx_val], X[idx_test]
            y_train, y_val, y_test = y[idx_train], y[idx_val], y[idx_test]

            model.fit(X_train, y_train)
            score_val = model.predict_proba(X_val)[:, 1]
            score_test = model.predict_proba(X_test)[:, 1]
            best = _best_threshold_by_f2(y_val, score_val)
            metrics = _metrics(y_test, score_test, best["threshold"])
            fold_metrics.append(metrics)

        df_metrics = pd.DataFrame(fold_metrics)
        row = {
            **params,
            "precision_mean": float(df_metrics["precision"].mean()),
            "precision_std": float(df_metrics["precision"].std(ddof=0)),
            "recall_mean": float(df_metrics["recall"].mean()),
            "recall_std": float(df_metrics["recall"].std(ddof=0)),
            "f1_mean": float(df_metrics["f1"].mean()),
            "f1_std": float(df_metrics["f1"].std(ddof=0)),
            "f2_mean": float(df_metrics["f2"].mean()),
            "f2_std": float(df_metrics["f2"].std(ddof=0)),
            "pr_auc_mean": float(df_metrics["pr_auc"].mean()),
            "pr_auc_std": float(df_metrics["pr_auc"].std(ddof=0)),
            "alert_rate_mean": float(df_metrics["alert_rate"].mean()),
            "alert_rate_std": float(df_metrics["alert_rate"].std(ddof=0)),
        }
        candidate_rows.append(row)
        evaluated += 1

        if row["f2_mean"] > best_f2:
            best_f2 = row["f2_mean"]
        if checkpoint_f2 is None:
            checkpoint_f2 = best_f2
            checkpoint_count = evaluated
        elif best_f2 - checkpoint_f2 >= 0.005:
            checkpoint_f2 = best_f2
            checkpoint_count = evaluated
        elif evaluated - checkpoint_count >= 20:
            early_stop_reason = "no_f2_improvement_20"
            break

    candidates_df = pd.DataFrame(candidate_rows)
    candidates_df.to_csv(run_dir / "metrics_candidates.csv", index=False)

    best_row = candidates_df.sort_values(["f2_mean", "recall_mean"], ascending=False).iloc[0].to_dict()
    best_params = {
        "n_estimators": int(best_row["n_estimators"]),
        "learning_rate": float(best_row["learning_rate"]),
        "max_depth": int(best_row["max_depth"]),
        "min_samples_leaf": int(best_row["min_samples_leaf"]),
        "subsample": float(best_row["subsample"]),
    }

    (run_dir / "best_config.json").write_text(json.dumps(best_row, indent=2), encoding="utf-8")

    val_scores_all = []
    val_labels_all = []
    test_scores_all = []
    test_labels_all = []
    test_rows = []
    importances = []

    for fold_idx, (idx_train, idx_val, idx_test) in enumerate(splits, start=1):
        model = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                (
                    "model",
                    GradientBoostingClassifier(
                        random_state=42,
                        n_estimators=best_params["n_estimators"],
                        learning_rate=best_params["learning_rate"],
                        max_depth=best_params["max_depth"],
                        min_samples_leaf=best_params["min_samples_leaf"],
                        subsample=best_params["subsample"],
                    ),
                ),
            ]
        )
        X_train, X_val, X_test = X[idx_train], X[idx_val], X[idx_test]
        y_train, y_val, y_test = y[idx_train], y[idx_val], y[idx_test]

        model.fit(X_train, y_train)
        score_val = model.predict_proba(X_val)[:, 1]
        score_test = model.predict_proba(X_test)[:, 1]

        val_scores_all.append(score_val)
        val_labels_all.append(y_val)
        test_scores_all.append(score_test)
        test_labels_all.append(y_test)

        model_step = model.named_steps["model"]
        if hasattr(model_step, "feature_importances_"):
            importances.append(model_step.feature_importances_)

        test_rows.append(
            pd.DataFrame(
                {
                    "index": idx_test,
                    "score": score_test,
                    "y_true": y_test,
                }
            )
        )

    val_scores_all = np.concatenate(val_scores_all)
    val_labels_all = np.concatenate(val_labels_all)
    test_scores_all = np.concatenate(test_scores_all)
    test_labels_all = np.concatenate(test_labels_all)

    thr_f2 = _best_threshold_by_f2(val_labels_all, val_scores_all)["threshold"]
    thr_hr = _threshold_high_recall(val_labels_all, val_scores_all, min_recall=0.90)
    thr_budget = _threshold_alert_budget(val_scores_all, target_rate=0.05)

    op_points = {}
    for name, thr in [("f2_opt", thr_f2), ("high_recall", thr_hr), ("alert_budget_5p", thr_budget)]:
        metrics = _metrics(test_labels_all, test_scores_all, thr)
        op_points[name] = {
            "threshold": float(thr),
            "precision": metrics["precision"],
            "recall": metrics["recall"],
            "f2": metrics["f2"],
            "alert_rate": metrics["alert_rate"],
            "alerts_per_1000": float(metrics["alert_rate"] * 1000.0),
        }

    (run_dir / "operating_points.json").write_text(json.dumps(op_points, indent=2), encoding="utf-8")

    scored_rows = pd.DataFrame(
        {
            "index": np.concatenate([r["index"].values for r in test_rows]),
            "anomaly_score": test_scores_all,
            "y_true": test_labels_all,
        }
    )
    scored_rows = scored_rows.groupby("index", as_index=False).agg(
        {"anomaly_score": "max", "y_true": "max"}
    )
    scored_rows["pred_label"] = scored_rows["anomaly_score"] >= thr_f2
    scored_rows["is_alert"] = scored_rows["pred_label"]

    merge_cols = []
    if "timestamp" in df_feat.columns:
        merge_cols.append("timestamp")
    merge_cols.extend([c for c in kpi_cols if c in df_feat.columns])
    if merge_cols:
        scored_rows = scored_rows.merge(
            df_feat[merge_cols], left_on="index", right_index=True, how="left"
        )
    scored_rows.to_csv(run_dir / "scored_labeled.csv", index=False)

    if importances:
        imp_arr = np.vstack(importances)
        imp_mean = imp_arr.mean(axis=0)
        imp_std = imp_arr.std(axis=0)
        imp_df = pd.DataFrame(
            {
                "feature": feature_cols,
                "importance_mean": imp_mean,
                "importance_std": imp_std,
            }
        ).sort_values("importance_mean", ascending=False)
        imp_df.to_csv(run_dir / "feature_importance.csv", index=False)
    else:
        imp_df = pd.DataFrame(columns=["feature", "importance_mean", "importance_std"])
        imp_df.to_csv(run_dir / "feature_importance.csv", index=False)

    test_rows_df = pd.concat(test_rows, ignore_index=True)
    test_rows_df["y_pred"] = (test_rows_df["score"] >= thr_f2).astype(int)
    df_examples = df_feat.copy()
    df_examples["index"] = df_examples.index
    joined = test_rows_df.merge(df_examples, on="index", how="left")

    tp_rows = joined[(joined["y_true"] == 1) & (joined["y_pred"] == 1)].sort_values("score", ascending=False).head(3)
    fp_rows = joined[(joined["y_true"] == 0) & (joined["y_pred"] == 1)].sort_values("score", ascending=False).head(3)

    def _format_examples(rows: pd.DataFrame) -> list[str]:
        examples = []
        for _, row in rows.iterrows():
            ts = row.get("timestamp")
            ts_str = ts.isoformat() if pd.notna(ts) else "n/a"
            top_feats = _top_feature_explanations(row, kpi_cols)
            if not top_feats:
                top_feats = ["n/a"]
            examples.append(
                f"- timestamp={ts_str}, score={row['score']:.4f}, top_features: " + "; ".join(top_feats)
            )
        return examples

    config_top10 = candidates_df.sort_values(["f2_mean", "recall_mean"], ascending=False).head(10)

    report_lines = []
    report_lines.append("# GradientBoosting time-CV tuning")
    report_lines.append("")
    report_lines.append("## Best configuration (by mean F2)")
    report_lines.append(json.dumps(best_params, indent=2))
    report_lines.append("")
    report_lines.append(f"Evaluated configs: {evaluated}")
    report_lines.append(f"Early stop: {early_stop_reason or 'none'}")
    report_lines.append("")
    report_lines.append("## Operating points (evaluated on test segments)")
    for key, val in op_points.items():
        report_lines.append(
            f"- {key}: threshold={val['threshold']:.6f}, precision={val['precision']:.4f}, "
            f"recall={val['recall']:.4f}, f2={val['f2']:.4f}, "
            f"alert_rate={val['alert_rate']:.4f}, alerts_per_1000={val['alerts_per_1000']:.1f}"
        )
    report_lines.append("")
    report_lines.append("## Top features (mean importance)")
    imp_top10 = imp_df.head(10) if not imp_df.empty else pd.DataFrame()
    if not imp_top10.empty:
        for _, row in imp_top10.iterrows():
            report_lines.append(f"- {row['feature']}: {row['importance_mean']:.4f} +/- {row['importance_std']:.4f}")
    else:
        report_lines.append("- n/a")
    report_lines.append("")
    report_lines.append("## Top 10 configs (mean F2)")
    report_lines.append("| n_estimators | learning_rate | max_depth | min_samples_leaf | subsample | f2_mean | recall_mean | pr_auc_mean | alert_rate_mean |")
    report_lines.append("|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for _, row in config_top10.iterrows():
        report_lines.append(
            f"| {int(row['n_estimators'])} | {row['learning_rate']:.2f} | {int(row['max_depth'])} | "
            f"{int(row['min_samples_leaf'])} | {row['subsample']:.2f} | {row['f2_mean']:.4f} | "
            f"{row['recall_mean']:.4f} | {row['pr_auc_mean']:.4f} | {row['alert_rate_mean']:.4f} |"
        )
    report_lines.append("")
    report_lines.append("## Example true positives")
    report_lines.extend(_format_examples(tp_rows))
    report_lines.append("")
    report_lines.append("## Example false positives")
    report_lines.extend(_format_examples(fp_rows))

    (run_dir / "report.md").write_text("\n".join(report_lines), encoding="utf-8")

    print(f"OK TimeCV tuning run saved to {run_dir}")


if __name__ == "__main__":
    main()
