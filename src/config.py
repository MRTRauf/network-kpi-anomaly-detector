from __future__ import annotations

NUM_COLS = ["bandwidth", "throughput", "congestion", "packet_loss", "latency", "jitter"]

                                                              
ROLL_WINDOW = 10

                                                                     
ALERT_QUANTILE = 0.99

TRAIN_RATIO = 0.70
VAL_RATIO = 0.15

                                    
IFOREST_PARAMS = {
    "n_estimators": 400,
    "contamination": 0.01,
    "random_state": 42,
    "n_jobs": 1,
}

RF_PARAMS = {
    "n_estimators": 400,
    "min_samples_leaf": 2,
    "class_weight": "balanced",
    "random_state": 42,
    "n_jobs": 1,
}

HGB_PARAMS = {
    "learning_rate": 0.1,
    "max_depth": 5,
    "max_iter": 300,
    "l2_regularization": 0.0,
    "random_state": 42,
}

CV_FOLDS = 3

                                                                                    
                                                       
INCIDENT_GAP_ROWS = 0
