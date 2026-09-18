# Industrial Predictive Maintenance and Failure-Risk Intelligence

An end-to-end, leakage-safe machine-learning project that converts operating conditions into cost-sensitive maintenance alerts. The repository goes beyond a single notebook: it includes data validation, physically interpretable feature engineering, model benchmarking, threshold optimization, held-out evaluation, explainability, automated tests, a model card, and a Streamlit application.

![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.6%2B-F7931E)
![License](https://img.shields.io/badge/code-MIT-green)
![Data](https://img.shields.io/badge/data-CC%20BY%204.0-blue)

## Business problem

Maintenance teams must balance two competing costs:

- **Missed failures** can create downtime, scrap, and safety risk.
- **False alarms** consume inspection and maintenance capacity.

This project estimates failure risk and selects an alert threshold from an explicit cost assumption instead of accepting the default probability threshold of 0.5.

## Dataset

The project uses the [AI4I 2020 Predictive Maintenance Dataset](https://archive.ics.uci.edu/dataset/601/ai4i+2020+predictive+maintenance+dataset) from UCI: 10,000 synthetic observations designed to reflect industrial predictive-maintenance conditions.

Model inputs:

- Product quality type
- Air temperature
- Process temperature
- Rotational speed
- Torque
- Tool wear

The five individual failure-mode columns are explicitly excluded from modelling because they encode the overall target and would create target leakage.

## Methodology

1. Validate schema, uniqueness, missingness, class balance, and target consistency.
2. Create physically interpretable temperature-difference, mechanical-power, and wear-torque features.
3. Use stratified 60/20/20 train, validation, and held-out test splits.
4. Benchmark Logistic Regression, Random Forest, and Histogram Gradient Boosting.
5. Select the model using validation average precision rather than misleading raw accuracy.
6. Select an operating threshold under a documented 25:1 false-negative/false-positive cost ratio.
7. Evaluate the frozen decision once on the held-out test set.
8. Produce permutation importance, curves, confusion matrix, predictions, tests, and an interactive application.

## Verified results

The committed pipeline was executed with random seed 42. Histogram Gradient Boosting achieved the strongest validation average precision and was selected before the held-out test set was opened.

| Held-out metric | Result |
|---|---:|
| ROC-AUC | 0.978 |
| Average precision | 0.902 |
| Recall | 0.912 |
| Precision | 0.348 |
| F1 | 0.504 |
| Brier score | 0.0087 |

The validation cost analysis selected a threshold of **0.02** under the documented 25:1 missed-failure/false-alarm cost ratio. On 2,000 held-out observations, the operating decision produced:

- 62 detected failures
- 6 missed failures
- 116 false alerts
- 1,816 correct non-failure decisions

The low threshold is intentional: under the stated cost assumption, the system prioritizes catching rare failures over minimizing inspections. It is not presented as a universal maintenance threshold.

Run `python run_pipeline.py` to reproduce the metrics. Detailed outputs are in:

- `reports/metrics.json`
- `reports/model_comparison.csv`
- `reports/threshold_analysis.csv`
- `reports/permutation_importance.csv`
- `reports/test_predictions.csv`

Permutation analysis identified torque, rotational speed, air temperature, process temperature, and tool wear as the five most influential raw inputs for held-out average precision.

## Repository structure

```text
industrial-predictive-maintenance-ml/
├── app/app.py
├── data/raw/ai4i2020.csv
├── models/failure_risk_model.joblib
├── notebooks/01_end_to_end_analysis.ipynb
├── reports/
│   ├── figures/
│   ├── metrics.json
│   ├── model_comparison.csv
│   └── threshold_analysis.csv
├── src/predictive_maintenance/
├── tests/
├── INTERVIEW_GUIDE.md
├── MODEL_CARD.md
├── RESUME_BULLETS.md
├── requirements.txt
└── run_pipeline.py
```

## Run locally

```bash
python -m venv .venv
```

Windows:

```bash
.venv\Scripts\activate
pip install -r requirements.txt
python run_pipeline.py
python -m unittest discover -s tests -v
streamlit run app/app.py
```

macOS/Linux:

```bash
source .venv/bin/activate
pip install -r requirements.txt
python run_pipeline.py
python -m unittest discover -s tests -v
streamlit run app/app.py
```

## Design decisions worth discussing in interviews

- Why failure-mode columns are leakage
- Why average precision is useful for rare failures
- Why the alert threshold is a business decision rather than a model constant
- Why the validation set chooses the model and threshold while the test set stays untouched
- Why synthetic benchmark performance cannot be treated as a plant-deployment result

## Responsible-use note

This is an educational portfolio project. The dataset is synthetic, the cost ratio is illustrative, and the model is not validated for safety-critical or production maintenance decisions.

## Author

**Burra Vijyusha**  
B.Tech, Metallurgical Engineering and Materials Science  
Indian Institute of Technology Indore
