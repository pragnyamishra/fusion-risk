"""Fusion Risk Streamlit dashboard.

Single-page app with four tabs:
  1. Predict        interactive scoring + top risk drivers (SHAP/LIME)
  2. Explainability  global SHAP + permutation importance + calibration curve
  3. Fairness Audit  Fairlearn disparate-impact across income groups
  4. Model Board     MLflow-tracked leaderboard across all model families

Run:  streamlit run app/streamlit_app.py
"""
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import config as C  # noqa: E402

st.set_page_config(page_title="Fusion Risk: Loan Default Prediction",
                   layout="wide")

# ----------------------------- styling -----------------------------
st.markdown("""
<style>
  .stApp { background-color: #0b0e14; color: #e6edf3; }
  h1, h2, h3 { color: #58a6ff; }
  div[data-testid="stMetricValue"] { color: #58a6ff; }
  /* let long metric values (e.g. model name) wrap instead of clipping to "..." */
  div[data-testid="stMetricValue"] {
      white-space: normal; overflow: visible; text-overflow: clip;
      font-size: 1.6rem; line-height: 1.2;
  }
  div[data-testid="stMetricValue"] > div { white-space: normal; overflow: visible; }
  .risk-high { color:#e74c3c; font-weight:700; }
  .risk-med  { color:#f39c12; font-weight:700; }
  .risk-low  { color:#2ecc71; font-weight:700; }

  /* keep content usable on small screens */
  .block-container { padding-left: 1rem; padding-right: 1rem; max-width: 1100px; }

  /* Mobile: stack every column row vertically so nothing gets squeezed */
  @media (max-width: 640px) {
    .block-container { padding-left: 0.6rem; padding-right: 0.6rem; padding-top: 1rem; }

    /* Streamlit horizontal blocks -> stack to full width */
    div[data-testid="stHorizontalBlock"] { flex-direction: column !important; gap: 0.25rem; }
    div[data-testid="stHorizontalBlock"] > div[data-testid="column"] {
        width: 100% !important; flex: 1 1 100% !important; min-width: 100% !important;
    }

    /* smaller metric text so 4-up metrics read cleanly when stacked */
    div[data-testid="stMetricValue"] { font-size: 1.4rem; }
    div[data-testid="stMetricLabel"] { font-size: 0.8rem; }

    h1 { font-size: 1.6rem; }
    h2 { font-size: 1.3rem; }
    h3 { font-size: 1.1rem; }

    /* tab labels wrap instead of overflowing */
    button[data-baseweb="tab"] { padding: 0.4rem 0.6rem; }
    div[data-testid="stTabs"] div[role="tablist"] { flex-wrap: wrap; gap: 0.25rem; }

    /* tables/dataframes scroll horizontally rather than clipping */
    div[data-testid="stDataFrame"] { overflow-x: auto; }
  }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_model():
    path = C.MODELS_DIR / "best_model.joblib"
    if not path.exists():
        return None, {}
    model = joblib.load(path)
    meta = {}
    mp = C.MODELS_DIR / "metadata.json"
    if mp.exists():
        meta = json.loads(mp.read_text())
    return model, meta


@st.cache_data
def load_reference():
    p = C.MODELS_DIR / "reference_sample.parquet"
    return pd.read_parquet(p) if p.exists() else None


def build_row(vals: dict) -> pd.DataFrame:
    inc = vals["annual_inc"] if vals["annual_inc"] else np.nan
    row = {
        "loan_amnt": vals["loan_amnt"], "loan_amnt_log": np.log1p(vals["loan_amnt"]),
        "int_rate": vals["int_rate"], "annual_inc": vals["annual_inc"],
        "income_log": np.log1p(vals["annual_inc"]), "dti": vals["dti"],
        "fico_range_high": vals["fico"], "installment": vals["installment"],
        "installment_to_income": vals["installment"] / inc if inc else 0.0,
        "loan_to_income": vals["loan_amnt"] / inc if inc else 0.0,
        "delinq_2yrs": vals["delinq"], "inq_last_6mths": vals["inq"],
        "pub_rec": vals["pub_rec"],
        "delinq_2yrs_log": np.log1p(vals["delinq"]),
        "inq_last_6mths_log": np.log1p(vals["inq"]),
        "pub_rec_log": np.log1p(vals["pub_rec"]),
        "behavioral_risk_score": (np.log1p(vals["delinq"]) * 2.0
                                  + np.log1p(vals["inq"]) * 1.5
                                  + np.log1p(vals["pub_rec"]) * 3.0),
        "emp_length_years": vals["emp_len"],
        "purpose": vals["purpose"], "home_ownership": vals["home"],
        "grade": vals["grade"],
        "text_combined": f"{vals['title']} {vals['purpose']} {vals['emp_title']}"
                          .lower().strip(),
    }
    return pd.DataFrame([row])


model, meta = load_model()
reference = load_reference()

st.title("Fusion Risk")
st.caption("Multimodal loan default prediction combining financial, behavioral, and text "
           "signals into a single production model.")

if model is None:
    st.error("No trained model found. Run `python -m src.train` and commit "
             "`models/best_model.joblib` before deploying.")
    st.stop()

best_name = meta.get("best_model", "model")
c1, c2, c3, c4 = st.columns(4)
lb = meta.get("leaderboard", [{}])
top = lb[0] if lb else {}
c1.metric("Best model", best_name)
c2.metric("ROC-AUC", f"{top.get('roc_auc', 0):.4f}")
c3.metric("Split", meta.get("split_kind", "n/a"))
c4.metric("Default rate", f"{meta.get('default_rate', 0)*100:.1f}%")

tab_pred, tab_explain, tab_fair, tab_board, tab_exp = st.tabs(
    ["Predict", "Explainability", "Fairness Audit", "Model Board", "Experiments"])

# ============================== PREDICT ==============================
with tab_pred:
    st.subheader("Score a loan application")
    col_a, col_b, col_c = st.columns(3)
    with col_a:
        loan_amnt = st.number_input("Loan amount ($)", 500, 40000, 15000, step=500)
        int_rate = st.slider("Interest rate (%)", 5.0, 31.0, 13.5, 0.1)
        annual_inc = st.number_input("Annual income ($)", 5000, 500000, 65000, step=1000)
        installment = st.number_input("Monthly installment ($)", 20, 2000, 450, step=10)
    with col_b:
        dti = st.slider("DTI ratio", 0.0, 50.0, 18.0, 0.5)
        fico = st.slider("FICO (range high)", 620, 850, 690, 1)
        emp_len = st.slider("Employment length (yrs)", 0.0, 10.0, 6.0, 0.5)
        grade = st.selectbox("Loan grade", list("ABCDEFG"), index=2)
    with col_c:
        purpose = st.selectbox("Purpose", [
            "debt_consolidation", "credit_card", "home_improvement", "major_purchase",
            "medical", "small_business", "car", "house", "moving", "vacation",
            "renewable_energy", "other"])
        home = st.selectbox("Home ownership", ["MORTGAGE", "RENT", "OWN", "OTHER"])
        delinq = st.number_input("Delinquencies (2yr)", 0, 20, 0)
        inq = st.number_input("Inquiries (6mo)", 0, 20, 1)
        pub_rec = st.number_input("Public records", 0, 20, 0)
    emp_title = st.text_input("Employment title", "teacher")
    title = st.text_input("Loan title", "Debt consolidation")

    if st.button("Predict default risk", type="primary"):
        X = build_row(dict(loan_amnt=loan_amnt, int_rate=int_rate, annual_inc=annual_inc,
                           installment=installment, dti=dti, fico=fico, emp_len=emp_len,
                           grade=grade, purpose=purpose, home=home, delinq=delinq,
                           inq=inq, pub_rec=pub_rec, emp_title=emp_title, title=title))
        proba = float(model.predict_proba(X)[:, 1][0])
        band = ("high", "risk-high") if proba >= 0.5 else \
               ("medium", "risk-med") if proba >= 0.25 else ("low", "risk-low")
        m1, m2 = st.columns([1, 2])
        m1.metric("Default probability", f"{proba*100:.1f}%")
        m1.markdown(f"Risk band: <span class='{band[1]}'>{band[0].upper()}</span>",
                    unsafe_allow_html=True)
        with m2:
            st.write("**Top risk drivers (SHAP)**")
            try:
                from src.explain import top_risk_drivers
                drivers = top_risk_drivers(model, X, k=5)
                dd = pd.DataFrame(drivers)
                dd["direction"] = np.where(dd["contribution"] >= 0,
                                           "increases risk", "lowers risk")
                st.dataframe(dd, hide_index=True, width="stretch")
            except Exception as e:
                st.info(f"Driver attribution unavailable: {e}")

# =========================== EXPLAINABILITY ===========================
with tab_explain:
    st.subheader("Global explainability")
    if reference is None:
        st.info("No reference sample found. Re-run training to generate it.")
    else:
        X_ref = reference.drop(columns=[C.TARGET])
        y_ref = reference[C.TARGET]
        colL, colR = st.columns(2)
        with colL:
            st.write("**SHAP mean absolute impact (top 15)**")
            try:
                from src.explain import shap_global
                imp = shap_global(model, X_ref).head(15)
                try:
                    st.bar_chart(imp.set_index("feature")["mean_abs_shap"],
                                 horizontal=True, height=400)
                except TypeError:
                    st.bar_chart(imp.set_index("feature")["mean_abs_shap"])
            except Exception as e:
                st.info(f"SHAP unavailable: {e}")
        with colR:
            st.write("**Permutation importance (top 15)**")
            try:
                from src.explain import permutation_importance_df
                pi = permutation_importance_df(model, X_ref, y_ref).head(15)
                try:
                    st.bar_chart(pi.set_index("feature")["importance"],
                                 horizontal=True, height=400)
                except TypeError:
                    st.bar_chart(pi.set_index("feature")["importance"])
            except Exception as e:
                st.info(f"Permutation importance unavailable: {e}")

        st.write("**Calibration curve**")
        try:
            from src.explain import calibration_data
            mp_, fp_ = calibration_data(model, X_ref, y_ref)
            cal = pd.DataFrame({"predicted": mp_, "observed": fp_}).set_index("predicted")
            st.line_chart(cal)
            st.caption("A perfectly calibrated model follows the diagonal. Points below "
                       "the diagonal mean the model is overestimating risk.")
        except Exception as e:
            st.info(f"Calibration unavailable: {e}")

# ============================== FAIRNESS ==============================
with tab_fair:
    st.subheader("Fairness audit across income groups (ECOA-oriented)")
    st.caption("Disparate-impact and error-rate parity checks via Fairlearn. "
               "Income group is a demonstrative sensitive attribute.")
    if reference is None:
        st.info("No reference sample available.")
    else:
        try:
            from src.fairness import audit
            table, summ = audit(model, reference.drop(columns=[C.TARGET]),
                                reference[C.TARGET], reference[C.SENSITIVE_COL])
            f1, f2, f3 = st.columns(3)
            di = summ["disparate_impact_ratio"]
            f1.metric("Disparate impact ratio", f"{di:.3f}",
                      help="Four-fifths rule: a ratio of 0.80 or above is the common "
                           "threshold for acceptable parity.")
            f2.metric("Passes 4/5 rule", "Yes" if summ["passes_four_fifths_rule"] else "No")
            f3.metric("Recall gap", f"{summ['recall_difference']:.3f}")
            st.dataframe(table, hide_index=True, width="stretch")
            st.json(summ)
        except Exception as e:
            st.info(f"Fairness audit unavailable: {e}")

# ============================== BOARD ==============================
with tab_board:
    st.subheader("Model leaderboard (MLflow-tracked)")
    if lb:
        board = pd.DataFrame(lb)[
            ["model", "roc_auc", "pr_auc", "f1", "precision", "recall"]]
        st.dataframe(board.style.format({c: "{:.4f}" for c in
                     ["roc_auc", "pr_auc", "f1", "precision", "recall"]}),
                     hide_index=True, width="stretch")
        st.caption("Every model run, including hyperparameters, metrics, and artifacts, "
                   "was tracked with MLflow during training for reproducible comparison "
                   "across all model families.")
    else:
        st.info("No leaderboard in metadata.")

# ============================== EXPERIMENTS ==============================
with tab_exp:
    st.subheader("MLflow experiment tracking")
    st.caption("Every training run was logged to MLflow. The runs below are read directly "
               "from the MLflow tracking store committed with this app, so the experiment "
               "history is visible here without launching a separate MLflow server.")
    try:
        from src.mlflow_reader import load_runs
        runs = load_runs()
    except Exception as e:
        runs = pd.DataFrame()
        st.info(f"Experiment store unavailable: {e}")

    if runs.empty:
        st.info("No MLflow runs found. The tracking store (mlruns/mlflow.db) is generated "
                "during training. Commit it alongside the model to display runs here.")
    else:
        metric_cols = ["roc_auc", "pr_auc", "f1", "precision", "recall"]
        display = runs.drop(columns=["_params"]).rename(columns={"run": "model"})
        st.write("**Tracked runs (sorted by ROC-AUC)**")
        st.dataframe(
            display.style.format({c: "{:.4f}" for c in metric_cols if c in display}),
            hide_index=True, width="stretch")

        st.write("**ROC-AUC by run**")
        chart_df = runs.set_index("run")["roc_auc"].dropna()
        try:
            st.bar_chart(chart_df, horizontal=True, height=360)
        except TypeError:
            st.bar_chart(chart_df)

        # Show the Optuna-tuned hyperparameters if a tuned run exists
        tuned = runs[runs["run"].str.contains("Optuna", case=False, na=False)]
        if not tuned.empty:
            params = tuned.iloc[0]["_params"]
            if params:
                st.write("**Best hyperparameters found by Optuna (XGBoost)**")
                pretty = {k: v for k, v in params.items() if k != "best_cv_roc_auc"}
                pdf = pd.DataFrame(
                    {"parameter": list(pretty.keys()),
                     "value": list(pretty.values())})
                st.dataframe(pdf, hide_index=True, width="stretch")
                if "best_cv_roc_auc" in params:
                    st.caption(f"Best cross-validated ROC-AUC during the Optuna search: "
                               f"{params['best_cv_roc_auc']}")

st.divider()
st.caption("Fusion Risk. FastAPI and Streamlit. XGBoost, LightGBM, CatBoost, "
           "Optuna, MLflow, SHAP, LIME, Fairlearn.")
