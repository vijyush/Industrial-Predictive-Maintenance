from __future__ import annotations

import numpy as np
import pandas as pd

from .config import BASE_FEATURES


def add_engineered_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Create physically interpretable features without using failure labels."""
    missing = sorted(set(BASE_FEATURES) - set(frame.columns))
    if missing:
        raise ValueError(f"Missing model inputs: {missing}")

    result = frame[BASE_FEATURES].copy()
    result["Temperature difference"] = (
        result["Process temperature"] - result["Air temperature"]
    )
    result["Mechanical power"] = (
        result["Torque"] * result["Rotational speed"] * 2.0 * np.pi / 60.0
    )
    result["Wear torque interaction"] = result["Tool wear"] * result["Torque"]
    return result

