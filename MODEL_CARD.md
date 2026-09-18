# Model Card: Machine-Failure Risk Classifier

## Intended use

This model is an educational decision-support demonstration for ranking operating observations by machine-failure risk. It is not validated for deployment on a real production line.

## Data

The model uses the AI4I 2020 synthetic predictive-maintenance dataset. The dataset contains 10,000 observations and six publishable operating inputs. The five individual failure-mode columns are excluded because they directly encode the target and would create leakage.

## Modelling approach

- Stratified 60/20/20 train, validation, and held-out test split
- Logistic Regression, Random Forest, and Histogram Gradient Boosting candidates
- Selection by validation average precision
- Cost-sensitive threshold using a 25:1 false-negative to false-positive cost ratio
- Held-out evaluation with ROC-AUC, average precision, precision, recall, F1, Brier score, and confusion matrix
- Model-agnostic permutation importance

## Verified held-out performance

- Selected model: Histogram Gradient Boosting
- Test observations: 2,000
- ROC-AUC: 0.978
- Average precision: 0.902
- Recall: 0.912
- Precision: 0.348
- F1: 0.504
- Brier score: 0.0087
- Confusion counts: TN 1,816; FP 116; FN 6; TP 62
- Selected alert threshold: 0.02 under the stated 25:1 cost ratio

These results describe this fixed split of a synthetic benchmark. They are not estimates of performance in an operating plant.

## Limitations

- The dataset is synthetic and does not represent a particular plant or asset.
- Observations are treated independently; equipment history and maintenance actions are unavailable.
- The selected business costs are illustrative, not estimated from a company.
- Dataset rules influence the labels, so excellent benchmark performance does not guarantee real-world transfer.
- Predictions must not be used as the sole basis for safety-critical decisions.

## Reproducibility

Run `python run_pipeline.py`. Verified metrics are written to `reports/metrics.json`, and the complete validation/test artifacts are stored in `reports/`.
