# Data

This project uses the **AI4I 2020 Predictive Maintenance Dataset** from the UCI Machine Learning Repository.

- Source: https://archive.ics.uci.edu/dataset/601/ai4i+2020+predictive+maintenance+dataset
- DOI: https://doi.org/10.24432/C5HS5C
- Size: 10,000 observations
- Task used here: binary classification of `Machine failure`
- Dataset license: Creative Commons Attribution 4.0 International (CC BY 4.0)

The original CSV is stored at `data/raw/ai4i2020.csv` for reproducibility. The software in this repository is separately released under the MIT License.

## Leakage warning

The source data also includes `TWF`, `HDF`, `PWF`, `OSF`, and `RNF`. These columns describe individual failure modes and directly determine or closely encode the overall failure label. They are retained only for descriptive validation and are **never used as model inputs**.

