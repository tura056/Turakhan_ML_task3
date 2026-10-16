# Task 3 ML Report — Bank Marketing Prediction

## 1. Framing & Preprocessing
- **Business Goal:** Predict bank deposit subscription (`y`).
- **10% Rule:** The call centre capacity allows calling only the top 10% customers with highest probability. Metric: ROC-AUC.
- **Train vs Test:** Column `duration` is present in `train.csv` but absent in `test.csv`. It was dropped during feature preparation.
- **Preprocessing:** Categoricals are imputed with `'unknown'` and One-Hot encoded. Numerics are median-imputed and standardized inside an `sklearn.Pipeline`.

## 2. Model Selection & Results
- **Baselines:** Majority class accuracy is 88.7% (ROC-AUC = 0.500), proving accuracy is misleading.
- **Models Evaluated:** LogisticRegression, RandomForest, GradientBoosting.
- **Tuned Model:** GradientBoosting achieved ROC-AUC = 0.79+ on CV.
- **10% Cutoff Performance:** Lift over random calls is ~3.5x.

## 3. Reproducibility
- Seed fixed to 42.
- Saved pipeline tested against raw `test.csv`.
