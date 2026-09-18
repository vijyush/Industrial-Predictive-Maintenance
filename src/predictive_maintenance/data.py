from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from .config import BASE_FEATURES, FAILURE_MODE_COLUMNS, ID_COLUMNS, TARGET


REQUIRED_COLUMNS = ID_COLUMNS + BASE_FEATURES + [TARGET] + FAILURE_MODE_COLUMNS


def load_data(path: str | Path) -> pd.DataFrame:
    """Load the official AI4I CSV and normalize optional unit suffixes."""
    frame = pd.read_csv(path)
    frame = frame.rename(
        columns={
            "Air temperature [K]": "Air temperature",
            "Process temperature [K]": "Process temperature",
            "Rotational speed [rpm]": "Rotational speed",
            "Torque [Nm]": "Torque",
            "Tool wear [min]": "Tool wear",
        }
    )
    return frame


def validate_data(frame: pd.DataFrame) -> dict[str, Any]:
    """Validate schema and return a transparent data-quality report."""
    missing_columns = sorted(set(REQUIRED_COLUMNS) - set(frame.columns))
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")

    if frame.empty:
        raise ValueError("Dataset is empty")
    if frame["UID"].duplicated().any():
        raise ValueError("UID must be unique")
    if not set(frame[TARGET].dropna().unique()).issubset({0, 1}):
        raise ValueError("Machine failure must be binary")
    if frame[REQUIRED_COLUMNS].isna().any().any():
        raise ValueError("Required columns contain missing values")

    mode_union = frame[FAILURE_MODE_COLUMNS].max(axis=1)
    mismatch_count = int((mode_union != frame[TARGET]).sum())

    return {
        "rows": int(len(frame)),
        "columns": int(frame.shape[1]),
        "duplicate_rows": int(frame.duplicated().sum()),
        "failure_count": int(frame[TARGET].sum()),
        "failure_rate": float(frame[TARGET].mean()),
        "failure_mode_target_mismatches": mismatch_count,
        "model_input_columns": BASE_FEATURES,
        "excluded_id_columns": ID_COLUMNS,
        "excluded_leakage_columns": FAILURE_MODE_COLUMNS,
    }


def modelling_frame(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Return leakage-safe raw predictors and the binary target."""
    return frame[BASE_FEATURES].copy(), frame[TARGET].astype(int).copy()

