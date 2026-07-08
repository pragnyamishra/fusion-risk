"""Data loading, cleaning, and multimodal feature engineering.

Reproduces the logic from the original research notebook and adds:
  - issue_d parsing for a temporal (time-based) train/test split
  - an income_group sensitive attribute for the fairness audit
"""
import gc
import os

import numpy as np
import pandas as pd

from src import config as C


def build_subset() -> pd.DataFrame:
    """Read the raw Kaggle CSV in chunks and cache a 100k clean subset."""
    if C.SUBSET_CSV.exists():
        print(f"Subset already exists: {C.SUBSET_CSV}")
        return pd.read_csv(C.SUBSET_CSV, low_memory=False)

    if not C.RAW_CSV.exists():
        raise FileNotFoundError(
            f"Raw CSV not found at {C.RAW_CSV}.\n"
            "Download accepted_2007_to_2018Q4.csv from "
            "https://www.kaggle.com/datasets/wordsforthewise/lending-club "
            f"and place it at {C.RAW_CSV}."
        )

    print(f"Creating subset of {C.ROWS_TO_LOAD:,} rows via chunked reading...")
    # Some raw files miss issue_d in older exports; read defensively.
    usecols = C.USECOLS
    chunks, total = [], 0
    for chunk in pd.read_csv(C.RAW_CSV, usecols=lambda c: c in usecols,
                             chunksize=50_000, low_memory=False):
        chunks.append(chunk)
        total += len(chunk)
        if total >= C.ROWS_TO_LOAD:
            break
    df = pd.concat(chunks).head(C.ROWS_TO_LOAD).copy()

    df["desc"] = df.get("desc", pd.Series([""] * len(df))).fillna("")
    df["emp_title"] = df.get("emp_title", pd.Series(["unknown"] * len(df))).fillna("unknown")
    df["title"] = df.get("title", pd.Series([""] * len(df))).fillna("")

    df.to_csv(C.SUBSET_CSV, index=False)
    print(f"Subset saved: {C.SUBSET_CSV} ({len(df):,} rows)")
    del chunks
    gc.collect()
    return df


def _parse_emp_len(x):
    if pd.isna(x) or x in ("", "nan"):
        return 0.0
    s = str(x).lower()
    if "10" in s:
        return 10.0
    if "<" in s:
        return 0.5
    digits = "".join(ch for ch in s if ch.isdigit())
    return float(digits) if digits else 0.0


def clean_and_engineer(df: pd.DataFrame) -> pd.DataFrame:
    """Full cleaning + three-modality feature engineering."""
    df = df.copy()

    # --- target ---
    df["int_rate"] = pd.to_numeric(
        df["int_rate"].astype(str).str.replace("%", "", regex=False), errors="coerce"
    )
    df[C.TARGET] = df["loan_status"].isin(C.DEFAULT_STATUSES).astype(np.int8)

    df.dropna(subset=["loan_amnt", "annual_inc", "dti", "int_rate", "fico_range_high"],
              inplace=True)

    # --- financial modality ---
    inc = df["annual_inc"].replace(0, np.nan)
    df["installment_to_income"] = df["installment"] / inc
    df["loan_to_income"] = df["loan_amnt"] / inc
    df["income_log"] = np.log1p(df["annual_inc"])
    df["loan_amnt_log"] = np.log1p(df["loan_amnt"])

    # --- employment ---
    df["emp_length_years"] = df["emp_length"].apply(_parse_emp_len)

    # --- behavioral modality ---
    for col in ["delinq_2yrs", "inq_last_6mths", "pub_rec"]:
        df[col] = pd.to_numeric(df.get(col), errors="coerce").fillna(0)
        df[col + "_log"] = np.log1p(df[col])
    df["behavioral_risk_score"] = (
        df["delinq_2yrs_log"] * 2.0
        + df["inq_last_6mths_log"] * 1.5
        + df["pub_rec_log"] * 3.0
    )

    # --- text modality ---
    df["text_combined"] = ""
    if "title" in df.columns:
        df["text_combined"] += df["title"].fillna("").astype(str) + " "
    df["text_combined"] += df["purpose"].fillna("").astype(str) + " "
    df["text_combined"] += df["emp_title"].fillna("").astype(str) + " "
    if "desc" in df.columns:
        meaningful = df["desc"].fillna("").astype(str).apply(
            lambda x: x if (len(str(x).split()) > 3 and x != "no description") else ""
        )
        df["text_combined"] += meaningful
    df["text_combined"] = df["text_combined"].str.lower().str.strip()

    # --- sensitive attribute for fairness audit (income tertiles) ---
    try:
        df[C.SENSITIVE_COL] = pd.qcut(
            df["annual_inc"], q=3, labels=["low_income", "mid_income", "high_income"]
        ).astype(str)
    except ValueError:
        df[C.SENSITIVE_COL] = "mid_income"

    # --- issue date for temporal split ---
    if "issue_d" in df.columns:
        df["issue_date"] = pd.to_datetime(df["issue_d"], format="%b-%Y", errors="coerce")
    else:
        df["issue_date"] = pd.NaT

    return df


def temporal_split(df: pd.DataFrame, test_frac: float = 0.25):
    """Time-based split: newest loans go to test to prevent leakage.

    Falls back to a stratified random split if issue_date is unavailable.
    """
    if df["issue_date"].notna().mean() > 0.8:
        df_sorted = df.sort_values("issue_date").reset_index(drop=True)
        cut = int(len(df_sorted) * (1 - test_frac))
        train, test = df_sorted.iloc[:cut], df_sorted.iloc[cut:]
        split_kind = "temporal"
    else:
        from sklearn.model_selection import train_test_split
        train, test = train_test_split(
            df, test_size=test_frac, stratify=df[C.TARGET], random_state=C.RANDOM_SEED
        )
        split_kind = "stratified_random_fallback"
    return train, test, split_kind


if __name__ == "__main__":
    raw = build_subset()
    eng = clean_and_engineer(raw)
    tr, te, kind = temporal_split(eng)
    print(f"Split={kind}  train={len(tr):,}  test={len(te):,}  "
          f"default_rate={eng[C.TARGET].mean():.3f}")
