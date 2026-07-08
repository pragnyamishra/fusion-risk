---
title: Fusion Risk
emoji: 📊
colorFrom: blue
colorTo: gray
sdk: docker
app_port: 7860
pinned: false
license: mit
---

# Fusion Risk — Multimodal Loan Default Prediction

A production-oriented, end-to-end credit-risk system built on the LendingClub
(2007–2018) dataset. It fuses three data modalities — **financial**, **behavioral**,
and **text** — into a single model, serves real-time predictions through a REST API and
an interactive dashboard, and ships with full explainability and a fairness audit.

**Live demo:** deployed on Hugging Face Spaces (Docker). The Streamlit dashboard is the
public UI; a FastAPI REST service runs alongside it in the same container.

---

## What it does

- **Multimodal fusion.** Financial ratios (loan-to-income, installment-to-income, log
  income), behavioral signals (delinquencies, inquiries, public records, composite risk
  score), one-hot categorical fields, and TF-IDF text features (employment title, loan
  purpose, title) are combined in one scikit-learn `ColumnTransformer` feeding a single
  classifier.
- **Model comparison.** Logistic Regression, Random Forest, Gradient Boosting, **XGBoost,
  LightGBM, and CatBoost** are trained and benchmarked on one leaderboard.
- **Hyperparameter tuning.** **Optuna** (TPE sampler) tunes XGBoost with stratified
  cross-validated ROC-AUC.
- **Leakage control.** A **temporal train/test split** (oldest loans train, newest test)
  prevents look-ahead leakage; falls back to a stratified split if issue dates are absent.
- **Experiment tracking.** **MLflow** logs every run's params, metrics, and the winning
  model for reproducible comparison.
- **Explainability.** Global **SHAP** and **permutation importance**, per-prediction
  **SHAP** drivers, **LIME** local explanations, and **calibration curves**.
- **Fairness.** A **Fairlearn** audit reports the disparate-impact ratio (four-fifths
  rule) and error-rate parity across income groups, in the spirit of **ECOA / Regulation
  B** responsible-lending standards.
- **Serving.** **FastAPI** exposes `/predict` returning a default probability, risk band,
  and top risk drivers; the whole stack is **Docker**-containerized and portable.

---

## Architecture

```
Raw LendingClub CSV
        │
        ▼
  Cleaning + Feature Engineering        (src/data.py)
  financial · behavioral · text · issue_d · income_group
        │
        ▼
  Temporal train/test split             (leakage-safe)
        │
        ▼
  ColumnTransformer                      (src/features.py)
  scale · one-hot · TF-IDF
        │
        ▼
  Train + compare + Optuna-tune          (src/train.py)
  LR · RF · GB · XGBoost · LightGBM · CatBoost   →  MLflow
        │
        ▼
  models/best_model.joblib  +  metadata.json  +  reference_sample.parquet
        │
        ├──────────────► FastAPI  /predict          (api/main.py)
        └──────────────► Streamlit dashboard         (app/streamlit_app.py)
                          Predict · Explain (SHAP/LIME/calibration) · Fairness · Board
```

---

## Run locally

```bash
# 1. install
pip install -r requirements.txt

# 2. add the dataset
#    download accepted_2007_to_2018Q4.csv from
#    https://www.kaggle.com/datasets/wordsforthewise/lending-club
#    and place it at data/accepted_2007_to_2018Q4.csv

# 3. train (creates models/best_model.joblib, metadata.json, reference_sample.parquet)
python -m src.train

# 4a. dashboard
streamlit run app/streamlit_app.py

# 4b. REST API
uvicorn api.main:app --reload --port 8000     # docs at http://localhost:8000/docs

# 5. inspect experiments
mlflow ui --backend-store-uri sqlite:///mlruns/mlflow.db
```

### Example API call

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"loan_amnt":20000,"int_rate":22.0,"annual_inc":45000,"dti":28.0,
       "fico_range_high":650,"installment":600,"emp_length_years":2.0,
       "delinq_2yrs":1,"inq_last_6mths":3,"pub_rec":1,
       "purpose":"small_business","home_ownership":"RENT","grade":"E",
       "emp_title":"driver","title":"Business loan"}'
```

Returns:

```json
{
  "default_probability": 0.41,
  "risk_band": "medium",
  "top_risk_drivers": [{"feature": "int_rate", "contribution": 0.46}, ...],
  "model_name": "XGBoost-Optuna"
}
```

---

## Deploy to Hugging Face Spaces

1. Train locally and commit the artifacts in `models/` (the raw CSV is git-ignored; the
   trained model is small and portable).
2. Create a new **Docker** Space and push this repo. Spaces builds the `Dockerfile`,
   launches Streamlit on port 7860, and gives you a public URL for your resume.

---

## Notes and limitations

- Data is 2007–2018 and pre-dates recent macro shifts; performance on modern lending may
  differ.
- Only approved LendingClub loans are included (no rejected applicants), limiting
  generalization to traditional bank lending.
- The income-group sensitive attribute is demonstrative. A real deployment would audit
  legally protected classes where permitted and pair the audit with mitigation.

## License

MIT
