from __future__ import annotations

import pandas as pd


def group_incidents(
    df: pd.DataFrame,
    alert_col: str = "is_alert",
    gap_rows: int = 0,
) -> pd.DataFrame:
    """
    Group consecutive alerts into incidents.

    Parameters
    ----------
    df : pd.DataFrame
        Must include 'timestamp' and alert_col. Assumes df already sorted by timestamp.
    alert_col : str
        Boolean column indicating an alert.
    gap_rows : int
        How many non-alert rows are allowed between alerts to still be merged into the same incident.
        - 0 = strictly consecutive alert rows.
        - 1 = allow one-row gap, etc.

    Returns
    -------
    pd.DataFrame
        Columns: incident_id, start_ts, end_ts, n_rows, max_score
    """
    if alert_col not in df.columns:
        raise ValueError(f"Missing alert column: {alert_col}")
    if "timestamp" not in df.columns:
        raise ValueError("Missing 'timestamp' column")

    alerts = df[df[alert_col] == True].copy()              
    if alerts.empty:
        return pd.DataFrame(columns=["incident_id", "start_ts", "end_ts", "n_rows", "max_score"])

    idx = alerts.index.to_numpy()

                                                                                      
    breaks = [0]
    for i in range(1, len(idx)):
        if (idx[i] - idx[i - 1]) > (gap_rows + 1):
            breaks.append(i)
    breaks.append(len(idx))

    rows = []
    incident_id = 0
    for b0, b1 in zip(breaks[:-1], breaks[1:]):
        chunk = alerts.iloc[b0:b1]
        rows.append(
            {
                "incident_id": incident_id,
                "start_ts": chunk["timestamp"].iloc[0],
                "end_ts": chunk["timestamp"].iloc[-1],
                "n_rows": int(len(chunk)),
                "max_score": float(chunk["anomaly_score"].max())
                if "anomaly_score" in chunk.columns
                else float("nan"),
            }
        )
        incident_id += 1

    return pd.DataFrame(rows)
