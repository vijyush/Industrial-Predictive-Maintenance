from __future__ import annotations

from pathlib import Path

import joblib
import pandas as pd

from .config import BASE_FEATURES, MODEL_PATH


def load_bundle(path: str | Path = MODEL_PATH) -> dict:
    bundle = joblib.load(path)
    required = {"pipeline", "threshold", "selected_model", "test_metrics"}
    missing = required - set(bundle)
    if missing:
        raise ValueError(f"Invalid model bundle; missing {sorted(missing)}")
    return bundle


def predict_record(record: dict, bundle: dict) -> dict:
    missing = sorted(set(BASE_FEATURES) - set(record))
    if missing:
        raise ValueError(f"Missing prediction inputs: {missing}")

    frame = pd.DataFrame([{name: record[name] for name in BASE_FEATURES}])
    probability = float(bundle["pipeline"].predict_proba(frame)[:, 1][0])
    threshold = float(bundle["threshold"])
    if probability >= threshold:
        band = "high"
    elif probability >= threshold * 0.5:
        band = "medium"
    else:
        band = "low"
    return {
        "failure_probability": probability,
        "decision_threshold": threshold,
        "maintenance_alert": bool(probability >= threshold),
        "risk_band": band,
        "model": bundle["selected_model"],
    }

