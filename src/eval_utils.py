from __future__ import annotations

import numpy as np
import pandas as pd


def time_sort(df: pd.DataFrame) -> pd.DataFrame:
    if "timestamp" not in df.columns:
        return df
    out = df.copy()
    out["timestamp"] = pd.to_datetime(out["timestamp"], errors="coerce", utc=True)
    out = out.dropna(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)
    return out


def time_split_indices(n: int, train_ratio: float, val_ratio: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    train_end = int(n * train_ratio)
    val_end = int(n * (train_ratio + val_ratio))
    idx = np.arange(n)
    return idx[:train_end], idx[train_end:val_end], idx[val_end:]


def precision_recall_at_k(y_true: np.ndarray, scores: np.ndarray, ks: list[float]) -> dict:
    n = len(y_true)
    order = np.argsort(scores)[::-1]
    out: dict = {}
    for k in ks:
        k_n = max(1, int(n * k))
        top_idx = order[:k_n]
        y_top = y_true[top_idx]
        precision = float(y_top.mean()) if len(y_top) else 0.0
        recall = float(y_top.sum() / max(1, y_true.sum()))
        out[f"precision_at_{int(k*100)}p"] = precision
        out[f"recall_at_{int(k*100)}p"] = recall
    return out
