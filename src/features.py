from __future__ import annotations

import pandas as pd
import numpy as np

from .config import ROLL_WINDOW


def _to_numeric_safe(s: pd.Series) -> pd.Series:
    s2 = pd.to_numeric(s, errors="coerce")
                                                                                           
    return s2.ffill().bfill()


def add_time_series_features(df: pd.DataFrame, num_cols: list[str]) -> pd.DataFrame:
    """
    Adds rolling features (mean/std) and 1-step diff for each numeric KPI.
    Rolling is row-based to be robust to uneven timestamp spacing.
    """
    out = df.copy()
    for c in num_cols:
        if c not in out.columns:
            raise ValueError(f"Missing expected numeric column: {c}")
        out[c] = _to_numeric_safe(out[c])

        out[f"{c}__diff1"] = out[c].diff().fillna(0.0)
        out[f"{c}__roll{ROLL_WINDOW}_mean"] = out[c].rolling(ROLL_WINDOW, min_periods=1).mean()
        out[f"{c}__roll{ROLL_WINDOW}_std"] = out[c].rolling(ROLL_WINDOW, min_periods=1).std().fillna(0.0)

    return out


def get_feature_columns(num_cols: list[str]) -> list[str]:
    cols: list[str] = []
    for c in num_cols:
        cols += [c, f"{c}__diff1", f"{c}__roll{ROLL_WINDOW}_mean", f"{c}__roll{ROLL_WINDOW}_std"]
    return cols


def per_row_contribution_z(df_feat: pd.DataFrame, num_cols: list[str]) -> pd.DataFrame:
    """
    Simple, model-agnostic "contribution" proxy:
    z-score per KPI using rolling mean/std. Higher absolute z suggests the KPI is unusual.
    Produces columns: contrib_<kpi>
    """
    out = df_feat.copy()
    eps = 1e-9
    for c in num_cols:
        mu = out[f"{c}__roll{ROLL_WINDOW}_mean"]
        sd = out[f"{c}__roll{ROLL_WINDOW}_std"].replace(0.0, eps)
        out[f"contrib_{c}"] = ((out[c] - mu) / sd).abs()
    return out
