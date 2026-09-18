import unittest
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from predictive_maintenance.modeling import choose_threshold, evaluate_binary, threshold_table


class ModelingTests(unittest.TestCase):
    def test_metrics_and_threshold_selection(self):
        y = np.array([0, 0, 0, 1, 1, 1])
        p = np.array([0.05, 0.20, 0.45, 0.40, 0.70, 0.95])
        metrics = evaluate_binary(y, p, threshold=0.5)
        self.assertGreater(metrics.roc_auc, 0.5)
        self.assertEqual(metrics.true_positives, 2)
        table = threshold_table(y, p, false_negative_cost=10, false_positive_cost=1)
        selected = choose_threshold(table)
        self.assertGreaterEqual(selected, 0.01)
        self.assertLessEqual(selected, 0.99)


if __name__ == "__main__":
    unittest.main()

