import unittest
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from predictive_maintenance.config import BASE_FEATURES, FAILURE_MODE_COLUMNS
from predictive_maintenance.data import modelling_frame, validate_data
from predictive_maintenance.features import add_engineered_features


def example_frame():
    return pd.DataFrame(
        {
            "UID": [1, 2],
            "Product ID": ["L1", "M2"],
            "Type": ["L", "M"],
            "Air temperature": [300.0, 301.0],
            "Process temperature": [310.0, 312.0],
            "Rotational speed": [1500, 1400],
            "Torque": [40.0, 45.0],
            "Tool wear": [100, 150],
            "Machine failure": [0, 1],
            "TWF": [0, 1],
            "HDF": [0, 0],
            "PWF": [0, 0],
            "OSF": [0, 0],
            "RNF": [0, 0],
        }
    )


class DataAndFeatureTests(unittest.TestCase):
    def test_validation_and_leakage_safe_frame(self):
        frame = example_frame()
        report = validate_data(frame)
        X, y = modelling_frame(frame)
        self.assertEqual(report["rows"], 2)
        self.assertEqual(X.columns.tolist(), BASE_FEATURES)
        self.assertNotIn("Machine failure", X.columns)
        self.assertTrue(set(FAILURE_MODE_COLUMNS).isdisjoint(X.columns))
        self.assertEqual(y.tolist(), [0, 1])

    def test_engineering_uses_physical_relations(self):
        X, _ = modelling_frame(example_frame())
        result = add_engineered_features(X)
        self.assertAlmostEqual(result.loc[0, "Temperature difference"], 10.0)
        expected_power = 40.0 * 1500 * 2 * np.pi / 60
        self.assertAlmostEqual(result.loc[0, "Mechanical power"], expected_power)
        self.assertAlmostEqual(result.loc[0, "Wear torque interaction"], 4000.0)

    def test_missing_feature_fails_fast(self):
        X, _ = modelling_frame(example_frame())
        with self.assertRaises(ValueError):
            add_engineered_features(X.drop(columns=["Torque"]))


if __name__ == "__main__":
    unittest.main()

