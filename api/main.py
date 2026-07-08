"""FastAPI service: real-time loan default probability + top risk drivers.

Run locally:  uvicorn api.main:app --reload --port 8000
Docs:         http://localhost:8000/docs
"""
import json
from pathlib import Path
from typing import Optional

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from src import config as C
from src.explain import top_risk_drivers

app = FastAPI(
    title="Fusion Risk API",
    description="Multimodal loan default prediction (financial + behavioral + text).",
    version="1.0.0",
)

_MODEL = None
_META = {}


def _load():
    global _MODEL, _META
    if _MODEL is None:
        path = C.MODELS_DIR / "best_model.joblib"
        if not path.exists():
            raise RuntimeError("Model not trained. Run: python -m src.train")
        _MODEL = joblib.load(path)
        meta_path = C.MODELS_DIR / "metadata.json"
        if meta_path.exists():
            _META = json.loads(meta_path.read_text())
    return _MODEL


class LoanApplication(BaseModel):
    loan_amnt: float = Field(..., example=15000)
    int_rate: float = Field(..., example=13.5)
    annual_inc: float = Field(..., example=65000)
    dti: float = Field(..., example=18.0)
    fico_range_high: float = Field(..., example=690)
    installment: float = Field(..., example=450)
    emp_length_years: float = Field(6.0, example=6.0)
    delinq_2yrs: float = Field(0, example=0)
    inq_last_6mths: float = Field(1, example=1)
    pub_rec: float = Field(0, example=0)
    purpose: str = Field("debt_consolidation", example="debt_consolidation")
    home_ownership: str = Field("MORTGAGE", example="MORTGAGE")
    grade: str = Field("C", example="C")
    emp_title: str = Field("unknown", example="teacher")
    title: str = Field("", example="Debt consolidation")


def _to_frame(app_in: LoanApplication) -> pd.DataFrame:
    import numpy as np
    d = app_in.dict()
    inc = d["annual_inc"] if d["annual_inc"] else np.nan
    row = {
        "loan_amnt": d["loan_amnt"], "loan_amnt_log": np.log1p(d["loan_amnt"]),
        "int_rate": d["int_rate"], "annual_inc": d["annual_inc"],
        "income_log": np.log1p(d["annual_inc"]), "dti": d["dti"],
        "fico_range_high": d["fico_range_high"], "installment": d["installment"],
        "installment_to_income": d["installment"] / inc if inc else 0.0,
        "loan_to_income": d["loan_amnt"] / inc if inc else 0.0,
        "delinq_2yrs": d["delinq_2yrs"], "inq_last_6mths": d["inq_last_6mths"],
        "pub_rec": d["pub_rec"],
        "delinq_2yrs_log": np.log1p(d["delinq_2yrs"]),
        "inq_last_6mths_log": np.log1p(d["inq_last_6mths"]),
        "pub_rec_log": np.log1p(d["pub_rec"]),
        "behavioral_risk_score": (np.log1p(d["delinq_2yrs"]) * 2.0
                                  + np.log1p(d["inq_last_6mths"]) * 1.5
                                  + np.log1p(d["pub_rec"]) * 3.0),
        "emp_length_years": d["emp_length_years"],
        "purpose": d["purpose"], "home_ownership": d["home_ownership"],
        "grade": d["grade"],
        "text_combined": f"{d['title']} {d['purpose']} {d['emp_title']}".lower().strip(),
    }
    return pd.DataFrame([row])


class PredictionResponse(BaseModel):
    default_probability: float
    risk_band: str
    top_risk_drivers: list
    model_name: Optional[str] = None


@app.get("/")
def root():
    return {"service": "Fusion Risk API", "status": "ok", "docs": "/docs"}


@app.get("/health")
def health():
    try:
        _load()
        return {"status": "healthy", "model": _META.get("best_model")}
    except Exception as e:
        raise HTTPException(status_code=503, detail=str(e))


@app.post("/predict", response_model=PredictionResponse)
def predict(app_in: LoanApplication):
    model = _load()
    X = _to_frame(app_in)
    proba = float(model.predict_proba(X)[:, 1][0])
    band = "high" if proba >= 0.5 else "medium" if proba >= 0.25 else "low"
    try:
        drivers = top_risk_drivers(model, X, k=5)
    except Exception:
        drivers = []
    return PredictionResponse(
        default_probability=round(proba, 4),
        risk_band=band,
        top_risk_drivers=drivers,
        model_name=_META.get("best_model"),
    )
