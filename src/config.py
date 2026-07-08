"""Central configuration for Fusion Risk."""
from pathlib import Path

RANDOM_SEED = 42
ROWS_TO_LOAD = 100_000

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
MODELS_DIR = ROOT / "models"
MLRUNS_DIR = ROOT / "mlruns"

DATA_DIR.mkdir(exist_ok=True)
MODELS_DIR.mkdir(exist_ok=True)

# Point this at your real Kaggle file. If missing, the pipeline errors with a clear message.
RAW_CSV = DATA_DIR / "accepted_2007_to_2018Q4.csv"
SUBSET_CSV = DATA_DIR / f"LendingClub_Subset{ROWS_TO_LOAD // 1000}k_clean.csv"

# Columns pulled from the raw LendingClub file
USECOLS = [
    "loan_amnt", "int_rate", "annual_inc", "dti", "fico_range_high",
    "emp_title", "purpose", "desc", "installment", "emp_length",
    "delinq_2yrs", "inq_last_6mths", "pub_rec", "loan_status",
    "title", "grade", "home_ownership", "verification_status",
    "issue_d",  # needed for temporal split
]

DEFAULT_STATUSES = ["Charged Off", "Default", "Late (31-120 days)", "Late (16-30 days)"]

TARGET = "default"

FINANCIAL_COLS = [
    "loan_amnt", "loan_amnt_log", "int_rate", "annual_inc", "income_log",
    "dti", "fico_range_high", "installment", "installment_to_income", "loan_to_income",
]
BEHAVIORAL_COLS = [
    "delinq_2yrs", "inq_last_6mths", "pub_rec",
    "delinq_2yrs_log", "inq_last_6mths_log", "pub_rec_log",
    "behavioral_risk_score", "emp_length_years",
]
CATEGORICAL_COLS = ["purpose", "home_ownership", "grade"]
TEXT_COL = "text_combined"

# Sensitive attribute for the fairness audit (derived from income bands)
SENSITIVE_COL = "income_group"

MLFLOW_EXPERIMENT = "fusion-risk-loan-default"
