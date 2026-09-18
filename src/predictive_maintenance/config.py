from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = PROJECT_ROOT / "data" / "raw" / "ai4i2020.csv"
MODEL_PATH = PROJECT_ROOT / "models" / "failure_risk_model.joblib"
REPORTS_DIR = PROJECT_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"

TARGET = "Machine failure"
ID_COLUMNS = ["UID", "Product ID"]
FAILURE_MODE_COLUMNS = ["TWF", "HDF", "PWF", "OSF", "RNF"]
BASE_FEATURES = [
    "Type",
    "Air temperature",
    "Process temperature",
    "Rotational speed",
    "Torque",
    "Tool wear",
]
NUMERIC_FEATURES = [
    "Air temperature",
    "Process temperature",
    "Rotational speed",
    "Torque",
    "Tool wear",
    "Temperature difference",
    "Mechanical power",
    "Wear torque interaction",
]
CATEGORICAL_FEATURES = ["Type"]

RANDOM_STATE = 42
FALSE_NEGATIVE_COST = 25.0
FALSE_POSITIVE_COST = 1.0

