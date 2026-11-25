# Disease Risk Prediction Using Explainable AI 

**Machine Learning course mini-project — Topic #5 (Healthcare & Social Impact)**

Predicting heart-disease risk from routine clinical measurements using **only classical Machine
Learning algorithms** (no deep learning), and explaining *why* each prediction is made.

## 1. Problem Statement & Objectives

Heart disease is a leading cause of death worldwide. Clinicians collect simple, inexpensive
measurements (age, blood pressure, cholesterol, ECG results, …), but combining many factors into a
reliable risk estimate is hard.

**Goal:** build an ML model that predicts whether a patient is at risk of heart disease from routine
measurements, and make its decisions transparent to a human.

Objectives:
1. Use a real, public clinical dataset (UCI Cleveland Heart Disease, 303 patients).
2. Preprocess (missing-value imputation, one-hot encoding, scaling) and explore the data (EDA).
3. Train and compare **6 classical algorithms**: Logistic Regression, Decision Tree, Random Forest,
   SVM (RBF), k-Nearest Neighbours, Gaussian Naive Bayes.
4. Evaluate with Accuracy, Precision, Recall, F1, ROC-AUC + **5-fold stratified cross-validation**;
   tune hyperparameters with grid search.
5. Explain predictions: odds ratios, feature importance, permutation importance, readable tree
   rules, and per-patient factor contributions.
6. Deliver a working prototype (in-notebook patient risk checker) and save all results.

## 2. Dataset

| | |
|---|---|
| **Name** | UCI Heart Disease — Cleveland (processed) |
| **Source** | https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data |
| **Size** | 303 patients × 13 features + binary target |
| **Target** | `0` = no disease (164), `1` = disease present (139) — original 0–4 severity binarised |
| **Missing** | `ca`: 4 values, `thal`: 2 values (imputed in the notebook) |

Included in this repo:
- `data/processed.cleveland.data` — original raw download (provenance)
- `data/heart_disease.csv` — clean CSV with column names and binary target (used by the notebook)

**Re-download / regenerate instructions** (if needed):

```bash
curl -L -o data/processed.cleveland.data \
  https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data
python3 prepare_data.py        # converts raw file -> data/heart_disease.csv
```

Feature dictionary: `age`, `sex` (1=male), `cp` (chest-pain type 1–4), `trestbps` (resting BP),
`chol` (cholesterol), `fbs` (fasting sugar > 120), `restecg` (resting ECG 0–2), `thalach`
(max heart rate), `exang` (exercise angina), `oldpeak` (exercise ST depression), `slope` (ST slope
1–3), `ca` (major vessels coloured 0–3), `thal` (3 normal / 6 fixed defect / 7 reversable defect).

## 3. Methodology

```
Raw UCI data ─► CSV + binary target
      │
      ▼
EDA (distributions, correlations, disease rates by cp/sex, boxplots)
      │
      ▼
Preprocessing: median/mode imputation ─► readable category names ─► one-hot encoding
               ─► stratified 80/20 split ─► StandardScaler inside pipelines (no leakage)
      │
      ▼
6 classical models ─► test metrics + 5-fold CV ─► GridSearchCV (RF, SVM)
      │
      ▼
Final model selected by CV ROC-AUC ─► XAI: odds ratios, importances, rules, local explanations
      │
      ▼
Bonus: PCA + K-Means patient segmentation ─► prototype risk checker ─► saved results/figures
```

Everything lives in **one executed Jupyter notebook**: `Disease_Risk_Prediction_XAI.ipynb`
(all outputs, tables and figures are already embedded — you can read it without running anything).

## 4. Experimental Results

Held-out test set (61 patients), all six classical models:

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC | CV Accuracy (5-fold) | CV ROC-AUC |
|---|---|---|---|---|---|---|---|
| **Random Forest** | **0.902** | 0.844 | **0.964** | **0.900** | **0.945** | 0.801 ± 0.035 | 0.885 |
| **Logistic Regression**  | 0.836 | 0.800 | 0.857 | 0.828 | 0.939 | **0.847 ± 0.017** | **0.905** |
| k-Nearest Neighbours | 0.836 | 0.765 | 0.929 | 0.839 | 0.922 | 0.797 ± 0.045 | 0.888 |
| SVM (RBF) | 0.820 | 0.815 | 0.786 | 0.800 | 0.912 | 0.826 ± 0.022 | 0.886 |
| Decision Tree | 0.787 | 0.742 | 0.821 | 0.780 | 0.864 | 0.736 ± 0.049 | 0.785 |
| Gaussian Naive Bayes | 0.590 | 0.532 | 0.893 | 0.667 | 0.847 | 0.658 ± 0.099 | 0.870 |

 **Selected final model: Logistic Regression** — best *cross-validated* ROC-AUC (0.905), the most
stable model (lowest CV variance), and inherently explainable. Grid-search-tuned Random Forest
(CV AUC 0.904) and SVM (0.898) were close runners-up; full table in `results/final_model_selection.csv`.

**Key observations**
- Random Forest achieved the best single test-split numbers (90.2% accuracy, 96.4% recall), but its
  CV score was lower and noisier — a textbook example of why model selection should use CV, not one split.
- In screening, **recall** matters most (missing a sick patient is costlier than a false alarm) —
  RF and kNN are strongest there.
- Naive Bayes struggles because the feature-independence assumption is badly violated (correlated
  clinical measurements) — its high recall comes with many false alarms.
- Explainability (all classical, no SHAP/deep learning): top risk drivers are `ca` (blocked vessels,
  **odds ratio ≈ 3.2** per SD), male `sex`, asymptomatic chest pain, `thal` defects, resting BP,
  low `thalach`, `oldpeak` — all clinically sensible.
- Bonus unsupervised check: K-Means (k=2) on scaled features splits patients into groups with
  **76% vs 14% disease prevalence** without ever seeing the target.

All figures: `results/figures/` (EDA, comparison bars, confusion matrices, ROC curves, coefficient
and importance plots, PCA/K-Means). Full tables: `results/*.csv`.

## 5. Working Prototype

Section 10 of the notebook is a small interactive risk checker:

```python
predict_patient(age=67, sex=1, cp=4, trestbps=160, chol=286, fbs=0, restecg=2,
                thalach=108, exang=1, oldpeak=1.5, slope=2, ca=3, thal=3)
# -> Predicted P(heart disease) = 99.8% -> HIGH RISK
#    Main risk-increasing factors: ca, thalach, trestbps
```

It prints the risk probability from the selected model plus the strongest increasing/protective
factors for that individual. The trained pipeline is saved to `results/best_model.joblib`
(load with `joblib.load`) so it can be wrapped in any app later.

## 6. How to Run

Requirements: Python ≥ 3.10.

```bash
pip install -r requirements.txt
jupyter notebook Disease_Risk_Prediction_XAI.ipynb   # then: Kernel -> Restart & Run All
```

or headless:

```bash
jupyter nbconvert --to notebook --execute --inplace Disease_Risk_Prediction_XAI.ipynb
```

The notebook reads `data/heart_disease.csv` (already included) and regenerates everything in
`results/`. Runtime ≈ 1 minute on a laptop.

## 7. Repository Structure

```
├── Disease_Risk_Prediction_XAI.ipynb   # complete project: code + executed results + figures
├── README.md
├── requirements.txt
├── prepare_data.py                     # raw UCI file -> clean CSV (reproducibility)
├── data/
│   ├── processed.cleveland.data        # original UCI download
│   └── heart_disease.csv               # dataset used by the notebook
└── results/
    ├── model_comparison.csv            # 6-model test + CV comparison
    ├── final_model_selection.csv       # incl. grid-search-tuned models
    ├── logistic_regression_explanation.csv  # coefficients + odds ratios
    ├── decision_tree_rules.txt         # human-readable rules
    ├── best_model.joblib               # trained final pipeline (prototype)
    └── figures/                        # all plots (PNG)
```

## 8. Limitations

- 303 patients, single centre, 1980s data → limited generalisation; small test set (61) → metric variance.
- Observational associations ≠ causation.
- Not calibrated for clinical deployment; no external validation.

## 9. Course Compliance

- Problem statement & objectives · public dataset · preprocessing + EDA
- 6 algorithms compared (≥ 3 required) · suitable metrics + CV · results & observations
- working prototype ·  **no deep learning — syllabus algorithms only**
  (regression, trees, ensembles, SVM, kNN, Naive Bayes, PCA, K-Means, grid search, standard metrics)