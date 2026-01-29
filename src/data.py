from __future__ import annotations

import pandas as pd


def load_csv(path: str) -> pd.DataFrame:
    """
    Load CSV and parse timestamps robustly.
    Keeps all columns; numeric conversion happens in feature pipeline.
    """
    df = pd.read_csv(path)
    if "timestamp" not in df.columns:
        raise ValueError("Expected a 'timestamp' column in the dataset.")
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce", utc=True)
    df = df.dropna(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)
    return df
