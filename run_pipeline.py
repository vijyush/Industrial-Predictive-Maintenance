from __future__ import annotations

import json
import sys
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.inspection import permutation_importance
from sklearn.metrics import ConfusionMatrixDisplay, PrecisionRecallDisplay, RocCurveDisplay
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from predictive_maintenance.config import (  # noqa: E402
    DATA_PATH,
    FALSE_NEGATIVE_COST,
    FALSE_POSITIVE_COST,
    FIGURES_DIR,
    MODEL_PATH,
    RANDOM_STATE,
    REPORTS_DIR,
)
from predictive_maintenance.data import load_data, modelling_frame, validate_data  # noqa: E402
from predictive_maintenance.modeling import (  # noqa: E402
    candidate_models,
    choose_threshold,
    evaluate_binary,
    threshold_table,
)


def save_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def setup_style() -> None:
    sns.set_theme(style="whitegrid", context="notebook")
    plt.rcParams.update({"figure.dpi": 130, "savefig.bbox": "tight"})


def main() -> None:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    setup_style()

    frame = load_data(DATA_PATH)
    quality = validate_data(frame)
    save_json(REPORTS_DIR / "data_validation.json", quality)
    X, y = modelling_frame(frame)

    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X, y, test_size=0.20, stratify=y, random_state=RANDOM_STATE
    )
    X_train, X_validation, y_train, y_validation = train_test_split(
        X_train_val,
        y_train_val,
        test_size=0.25,
        stratify=y_train_val,
        random_state=RANDOM_STATE,
    )

    comparison_rows = []
    fitted_models = {}
    validation_probabilities = {}
    for name, pipeline in candidate_models().items():
        pipeline.fit(X_train, y_train)
        probabilities = pipeline.predict_proba(X_validation)[:, 1]
        metrics = evaluate_binary(y_validation, probabilities, threshold=0.5)
        comparison_rows.append({"model": name, **metrics.to_dict()})
        fitted_models[name] = pipeline
        validation_probabilities[name] = probabilities

    comparison = pd.DataFrame(comparison_rows).sort_values(
        ["average_precision", "roc_auc"], ascending=False
    )
    comparison.to_csv(REPORTS_DIR / "model_comparison.csv", index=False)
    selected_name = str(comparison.iloc[0]["model"])
    selected_model = fitted_models[selected_name]

    thresholds = threshold_table(
        y_validation,
        validation_probabilities[selected_name],
        false_negative_cost=FALSE_NEGATIVE_COST,
        false_positive_cost=FALSE_POSITIVE_COST,
    )
    thresholds.to_csv(REPORTS_DIR / "threshold_analysis.csv", index=False)
    selected_threshold = choose_threshold(thresholds)

    test_probabilities = selected_model.predict_proba(X_test)[:, 1]
    test_metrics = evaluate_binary(y_test, test_probabilities, selected_threshold)

    importance = permutation_importance(
        selected_model,
        X_test,
        y_test,
        scoring="average_precision",
        n_repeats=12,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    importance_table = pd.DataFrame(
        {
            "feature": X_test.columns,
            "importance_mean": importance.importances_mean,
            "importance_std": importance.importances_std,
        }
    ).sort_values("importance_mean", ascending=False)
    importance_table.to_csv(REPORTS_DIR / "permutation_importance.csv", index=False)

    predictions = X_test.copy()
    predictions["actual_failure"] = y_test.values
    predictions["failure_probability"] = test_probabilities
    predictions["predicted_alert"] = (test_probabilities >= selected_threshold).astype(int)
    predictions.to_csv(REPORTS_DIR / "test_predictions.csv", index=False)

    metrics_payload = {
        "dataset": "AI4I 2020 Predictive Maintenance",
        "rows": quality["rows"],
        "failure_rate": quality["failure_rate"],
        "split_rows": {
            "train": int(len(X_train)),
            "validation": int(len(X_validation)),
            "test": int(len(X_test)),
        },
        "selection_metric": "validation average precision",
        "selected_model": selected_name,
        "selected_threshold": selected_threshold,
        "cost_assumptions": {
            "false_negative": FALSE_NEGATIVE_COST,
            "false_positive": FALSE_POSITIVE_COST,
        },
        "test_metrics": test_metrics.to_dict(),
        "top_permutation_features": importance_table.head(5).to_dict("records"),
    }
    save_json(REPORTS_DIR / "metrics.json", metrics_payload)

    bundle = {
        "pipeline": selected_model,
        "threshold": selected_threshold,
        "selected_model": selected_name,
        "test_metrics": test_metrics.to_dict(),
        "cost_assumptions": metrics_payload["cost_assumptions"],
        "dataset": metrics_payload["dataset"],
    }
    joblib.dump(bundle, MODEL_PATH)

    # Figure 1: class balance
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    counts = y.value_counts().sort_index()
    sns.barplot(x=["No failure", "Failure"], y=counts.values, ax=ax, color="#31688e")
    ax.set_title("AI4I target distribution")
    ax.set_ylabel("Observations")
    for index, value in enumerate(counts.values):
        ax.text(index, value, f"{value:,}", ha="center", va="bottom")
    fig.savefig(FIGURES_DIR / "class_balance.png")
    plt.close(fig)

    # Figure 2: model comparison
    melted = comparison.melt(
        id_vars="model",
        value_vars=["average_precision", "roc_auc", "f1"],
        var_name="metric",
        value_name="score",
    )
    fig, ax = plt.subplots(figsize=(8.2, 4.8))
    sns.barplot(data=melted, x="model", y="score", hue="metric", ax=ax)
    ax.set_ylim(0, 1)
    ax.set_xlabel("")
    ax.set_title("Validation-set model comparison")
    ax.tick_params(axis="x", rotation=10)
    fig.savefig(FIGURES_DIR / "model_comparison.png")
    plt.close(fig)

    # Figure 3: held-out curves
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.4))
    PrecisionRecallDisplay.from_predictions(y_test, test_probabilities, ax=axes[0])
    axes[0].set_title("Held-out precision-recall curve")
    RocCurveDisplay.from_predictions(y_test, test_probabilities, ax=axes[1])
    axes[1].set_title("Held-out ROC curve")
    fig.savefig(FIGURES_DIR / "heldout_curves.png")
    plt.close(fig)

    # Figure 4: threshold cost
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    ax.plot(thresholds["threshold"], thresholds["cost_per_1000"], color="#d95f02")
    ax.axvline(selected_threshold, color="#1b9e77", linestyle="--", label=f"Selected {selected_threshold:.2f}")
    ax.set_title("Validation cost by alert threshold")
    ax.set_xlabel("Failure-risk threshold")
    ax.set_ylabel("Assumed cost per 1,000 observations")
    ax.legend()
    fig.savefig(FIGURES_DIR / "threshold_cost.png")
    plt.close(fig)

    # Figure 5: confusion matrix
    fig, ax = plt.subplots(figsize=(5.2, 4.6))
    ConfusionMatrixDisplay.from_predictions(
        y_test,
        (test_probabilities >= selected_threshold).astype(int),
        display_labels=["No failure", "Failure"],
        cmap="Blues",
        ax=ax,
    )
    ax.set_title(f"Held-out decisions at threshold {selected_threshold:.2f}")
    fig.savefig(FIGURES_DIR / "confusion_matrix.png")
    plt.close(fig)

    # Figure 6: model-agnostic importance
    top = importance_table.head(8).sort_values("importance_mean")
    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    ax.barh(top["feature"], top["importance_mean"], xerr=top["importance_std"], color="#35b779")
    ax.set_title("Permutation importance on held-out data")
    ax.set_xlabel("Decrease in average precision")
    fig.savefig(FIGURES_DIR / "permutation_importance.png")
    plt.close(fig)

    print(json.dumps(metrics_payload, indent=2))


if __name__ == "__main__":
    main()

