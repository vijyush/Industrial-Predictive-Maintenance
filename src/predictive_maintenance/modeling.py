from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler

from .config import (
    CATEGORICAL_FEATURES,
    FALSE_NEGATIVE_COST,
    FALSE_POSITIVE_COST,
    NUMERIC_FEATURES,
    RANDOM_STATE,
)
from .features import add_engineered_features


@dataclass
class BinaryMetrics:
    roc_auc: float
    average_precision: float
    precision: float
    recall: float
    f1: float
    brier_score: float
    true_negatives: int
    false_positives: int
    false_negatives: int
    true_positives: int
    threshold: float

    def to_dict(self) -> dict:
        return asdict(self)


def _preprocessor() -> ColumnTransformer:
    numeric = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )
    return ColumnTransformer(
        [("numeric", numeric, NUMERIC_FEATURES), ("categorical", categorical, CATEGORICAL_FEATURES)],
        remainder="drop",
    )


def candidate_models() -> dict[str, Pipeline]:
    classifiers = {
        "logistic_regression": LogisticRegression(
            C=0.5,
            class_weight="balanced",
            max_iter=2500,
            random_state=RANDOM_STATE,
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=450,
            max_depth=9,
            min_samples_leaf=3,
            class_weight="balanced_subsample",
            n_jobs=-1,
            random_state=RANDOM_STATE,
        ),
        "hist_gradient_boosting": HistGradientBoostingClassifier(
            learning_rate=0.05,
            max_iter=300,
            max_leaf_nodes=15,
            l2_regularization=1.0,
            class_weight="balanced",
            random_state=RANDOM_STATE,
        ),
    }
    return {
        name: Pipeline(
            [
                ("feature_engineering", FunctionTransformer(add_engineered_features, validate=False)),
                ("preprocess", _preprocessor()),
                ("classifier", classifier),
            ]
        )
        for name, classifier in classifiers.items()
    }


def evaluate_binary(y_true, probabilities, threshold: float = 0.5) -> BinaryMetrics:
    probabilities = np.asarray(probabilities, dtype=float)
    predictions = (probabilities >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, predictions, labels=[0, 1]).ravel()
    return BinaryMetrics(
        roc_auc=float(roc_auc_score(y_true, probabilities)),
        average_precision=float(average_precision_score(y_true, probabilities)),
        precision=float(precision_score(y_true, predictions, zero_division=0)),
        recall=float(recall_score(y_true, predictions, zero_division=0)),
        f1=float(f1_score(y_true, predictions, zero_division=0)),
        brier_score=float(brier_score_loss(y_true, probabilities)),
        true_negatives=int(tn),
        false_positives=int(fp),
        false_negatives=int(fn),
        true_positives=int(tp),
        threshold=float(threshold),
    )


def threshold_table(
    y_true,
    probabilities,
    false_negative_cost: float = FALSE_NEGATIVE_COST,
    false_positive_cost: float = FALSE_POSITIVE_COST,
) -> pd.DataFrame:
    rows = []
    for threshold in np.linspace(0.01, 0.99, 99):
        metrics = evaluate_binary(y_true, probabilities, float(threshold))
        cost = (
            metrics.false_negatives * false_negative_cost
            + metrics.false_positives * false_positive_cost
        )
        rows.append(
            {
                **metrics.to_dict(),
                "expected_cost": float(cost),
                "cost_per_1000": float(cost / len(y_true) * 1000.0),
            }
        )
    return pd.DataFrame(rows)


def choose_threshold(table: pd.DataFrame) -> float:
    best = table.sort_values(
        ["expected_cost", "recall", "precision"], ascending=[True, False, False]
    ).iloc[0]
    return float(best["threshold"])

