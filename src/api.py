from __future__ import annotations

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from .config import NUM_COLS, ROLL_WINDOW
from .features import add_time_series_features, get_feature_columns, per_row_contribution_z
from .run_utils import latest_run_dir

ART_DIR = "artifacts"

app = FastAPI(title="Network KPI Anomaly Scoring API", version="1.0.0")


class ScoreRequest(BaseModel):
    """
    Provide the latest window of KPI rows (recommended >= ROLL_WINDOW).
    The API returns per-row anomaly scores and alerts (threshold from artifacts/meta.joblib).
    """
    rows: list[dict] = Field(..., description="List of rows, each must include timestamp and KPI columns.")


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/score")
def score(req: ScoreRequest):
    run_dir = latest_run_dir("train")
    if run_dir is None:
        raise HTTPException(status_code=500, detail="No training run found. Run training first.")
    try:
        meta = joblib.load(run_dir / "meta.joblib")
        model = joblib.load(run_dir / "model.joblib")
        scaler = joblib.load(run_dir / "scaler.joblib")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Artifacts missing in {run_dir}. Details: {e}")

    df = pd.DataFrame(req.rows)
    if "timestamp" not in df.columns:
        raise HTTPException(status_code=400, detail="Each row must include 'timestamp'.")

    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce", utc=True)
    df = df.dropna(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)

                              
    for c in NUM_COLS:
        if c not in df.columns:
            raise HTTPException(status_code=400, detail=f"Missing KPI column: {c}")

    df_feat = add_time_series_features(df, NUM_COLS)
    df_feat = per_row_contribution_z(df_feat, NUM_COLS)

    feat_cols = meta["feature_cols"]
    X = df_feat[feat_cols].astype(float).values

    model_type = meta.get("model_type", "IsolationForest")
    if model_type == "IsolationForest":
        Xs = scaler.transform(X)
        scores = -model.decision_function(Xs)
    else:
        scores = model.predict_proba(X)[:, 1]
    thr = float(meta["threshold"])
    df_feat["anomaly_score"] = scores
    df_feat["is_alert"] = df_feat["anomaly_score"] >= thr

                                                                          
    last = df_feat.iloc[-1].to_dict()
    contrib_cols = [f"contrib_{c}" for c in NUM_COLS]
    contrib = {c: float(last.get(c, 0.0)) for c in contrib_cols}
    top = sorted(contrib.items(), key=lambda kv: kv[1], reverse=True)[:3]

    return {
        "threshold": thr,
        "window_rows_used": int(len(df_feat)),
        "latest": {
            "timestamp": str(last["timestamp"]),
            "anomaly_score": float(last["anomaly_score"]),
            "is_alert": bool(last["is_alert"]),
            "top_contributors": top,
        },
    }
