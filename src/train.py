"""Train and compare all models, tune with Optuna, log everything to MLflow.

Run:  python -m src.train
Produces:
  models/best_model.joblib      (full fitted pipeline: preprocessor + classifier)
  models/metadata.json          (metrics, feature list, split info, model name)
  mlruns/                       (MLflow experiment tracking)
"""
import json
import time
import warnings

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (average_precision_score, f1_score,
                             precision_score, recall_score, roc_auc_score)
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline

import mlflow
import optuna
from optuna.samplers import TPESampler

from src import config as C
from src.data import build_subset, clean_and_engineer, temporal_split
from src.features import build_preprocessor

warnings.filterwarnings("ignore")
optuna.logging.set_verbosity(optuna.logging.WARNING)

try:
    from xgboost import XGBClassifier
    HAS_XGB = True
except ImportError:
    HAS_XGB = False
try:
    from lightgbm import LGBMClassifier
    HAS_LGBM = True
except ImportError:
    HAS_LGBM = False
try:
    from catboost import CatBoostClassifier
    HAS_CAT = True
except ImportError:
    HAS_CAT = False


def _metrics(y_true, proba, thresh=0.5):
    pred = (proba >= thresh).astype(int)
    return {
        "roc_auc": roc_auc_score(y_true, proba),
        "pr_auc": average_precision_score(y_true, proba),
        "f1": f1_score(y_true, pred, zero_division=0),
        "precision": precision_score(y_true, pred, zero_division=0),
        "recall": recall_score(y_true, pred, zero_division=0),
    }


def _pipe(classifier):
    return Pipeline([("preprocessor", build_preprocessor()),
                     ("classifier", classifier)])


def _scale_pos_weight(y):
    neg, pos = (y == 0).sum(), (y == 1).sum()
    return neg / max(pos, 1)


def build_candidates(y_train):
    spw = _scale_pos_weight(y_train)
    c = {
        "LogisticRegression": LogisticRegression(
            max_iter=500, class_weight="balanced", solver="saga",
            random_state=C.RANDOM_SEED, n_jobs=-1),
        "RandomForest": RandomForestClassifier(
            n_estimators=200, max_depth=15, min_samples_split=10,
            class_weight="balanced_subsample", random_state=C.RANDOM_SEED, n_jobs=-1),
        "GradientBoosting": GradientBoostingClassifier(
            n_estimators=100, max_depth=5, learning_rate=0.1,
            subsample=0.8, random_state=C.RANDOM_SEED),
    }
    if HAS_XGB:
        c["XGBoost"] = XGBClassifier(
            n_estimators=300, max_depth=5, learning_rate=0.05, subsample=0.8,
            colsample_bytree=0.8, scale_pos_weight=spw, eval_metric="auc",
            tree_method="hist", random_state=C.RANDOM_SEED, n_jobs=-1)
    if HAS_LGBM:
        c["LightGBM"] = LGBMClassifier(
            n_estimators=400, max_depth=-1, num_leaves=63, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8, scale_pos_weight=spw,
            random_state=C.RANDOM_SEED, n_jobs=-1, verbose=-1)
    if HAS_CAT:
        c["CatBoost"] = CatBoostClassifier(
            iterations=400, depth=6, learning_rate=0.05, l2_leaf_reg=3.0,
            scale_pos_weight=spw, random_seed=C.RANDOM_SEED, verbose=0)
    return c


def optuna_tune_xgb(X_train, y_train, n_trials=25):
    """Optuna TPE search over XGBoost, scored with stratified CV ROC-AUC."""
    if not HAS_XGB:
        return None, None
    spw = _scale_pos_weight(y_train)
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=C.RANDOM_SEED)

    def objective(trial):
        params = dict(
            n_estimators=trial.suggest_int("n_estimators", 200, 500, step=50),
            max_depth=trial.suggest_int("max_depth", 3, 8),
            learning_rate=trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
            subsample=trial.suggest_float("subsample", 0.6, 1.0),
            colsample_bytree=trial.suggest_float("colsample_bytree", 0.6, 1.0),
            min_child_weight=trial.suggest_int("min_child_weight", 1, 10),
            gamma=trial.suggest_float("gamma", 0.0, 5.0),
        )
        clf = XGBClassifier(**params, scale_pos_weight=spw, eval_metric="auc",
                            tree_method="hist", random_state=C.RANDOM_SEED, n_jobs=-1)
        scores = cross_val_score(_pipe(clf), X_train, y_train, cv=cv,
                                 scoring="roc_auc", n_jobs=-1)
        return scores.mean()

    study = optuna.create_study(direction="maximize",
                                sampler=TPESampler(seed=C.RANDOM_SEED))
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    best = XGBClassifier(**study.best_params, scale_pos_weight=spw, eval_metric="auc",
                         tree_method="hist", random_state=C.RANDOM_SEED, n_jobs=-1)
    return _pipe(best), study


def main(n_trials=25):
    print("Loading + engineering features...")
    df = clean_and_engineer(build_subset())
    train_df, test_df, split_kind = temporal_split(df)
    X_train, y_train = train_df.drop(columns=[C.TARGET]), train_df[C.TARGET]
    X_test, y_test = test_df.drop(columns=[C.TARGET]), test_df[C.TARGET]
    print(f"Split={split_kind}  train={len(X_train):,}  test={len(X_test):,}")

    C.MLRUNS_DIR.mkdir(exist_ok=True)
    mlflow.set_tracking_uri(f"sqlite:///{C.MLRUNS_DIR / 'mlflow.db'}")
    mlflow.set_experiment(C.MLFLOW_EXPERIMENT)

    results, fitted = [], {}
    candidates = build_candidates(y_train)

    for name, clf in candidates.items():
        with mlflow.start_run(run_name=name):
            t0 = time.time()
            pipe = _pipe(clf)
            pipe.fit(X_train, y_train)
            train_time = time.time() - t0
            proba = pipe.predict_proba(X_test)[:, 1]
            m = _metrics(y_test, proba)
            m.update({"model": name, "train_time_s": round(train_time, 2)})
            mlflow.log_param("model_family", name)
            mlflow.log_param("split_kind", split_kind)
            mlflow.log_metrics({k: v for k, v in m.items()
                                if isinstance(v, (int, float))})
            results.append(m)
            fitted[name] = pipe
            print(f"  {name:18s} ROC-AUC={m['roc_auc']:.4f} F1={m['f1']:.3f} "
                  f"recall={m['recall']:.3f} ({train_time:.1f}s)")

    # --- Optuna-tuned XGBoost ---
    if HAS_XGB and n_trials > 0:
        print(f"\nOptuna tuning XGBoost ({n_trials} trials)...")
        with mlflow.start_run(run_name="XGBoost-Optuna"):
            tuned_pipe, study = optuna_tune_xgb(X_train, y_train, n_trials)
            tuned_pipe.fit(X_train, y_train)
            proba = tuned_pipe.predict_proba(X_test)[:, 1]
            m = _metrics(y_test, proba)
            m.update({"model": "XGBoost-Optuna", "train_time_s": None})
            mlflow.log_params(study.best_params)
            mlflow.log_param("best_cv_roc_auc", round(study.best_value, 4))
            mlflow.log_metrics({k: v for k, v in m.items()
                                if isinstance(v, (int, float))})
            results.append(m)
            fitted["XGBoost-Optuna"] = tuned_pipe
            print(f"  best CV ROC-AUC={study.best_value:.4f}  "
                  f"test ROC-AUC={m['roc_auc']:.4f}")

    res_df = pd.DataFrame(results).sort_values("roc_auc", ascending=False)
    print("\n=== Leaderboard ===")
    print(res_df[["model", "roc_auc", "pr_auc", "f1", "precision",
                  "recall"]].to_string(index=False))

    best_name = res_df.iloc[0]["model"]
    best_pipe = fitted[best_name]
    joblib.dump(best_pipe, C.MODELS_DIR / "best_model.joblib")

    meta = {
        "best_model": best_name,
        "split_kind": split_kind,
        "default_rate": float(df[C.TARGET].mean()),
        "n_train": int(len(X_train)),
        "n_test": int(len(X_test)),
        "leaderboard": res_df.to_dict(orient="records"),
        "features": {
            "financial": C.FINANCIAL_COLS,
            "behavioral": C.BEHAVIORAL_COLS,
            "categorical": C.CATEGORICAL_COLS,
            "text": C.TEXT_COL,
        },
        "libs": {"xgboost": HAS_XGB, "lightgbm": HAS_LGBM, "catboost": HAS_CAT},
    }
    (C.MODELS_DIR / "metadata.json").write_text(json.dumps(meta, indent=2, default=str))
    # Save a small reference sample for SHAP background + fairness audit in the app
    test_df.sample(min(2000, len(test_df)), random_state=C.RANDOM_SEED)\
        .to_parquet(C.MODELS_DIR / "reference_sample.parquet")
    print(f"\nSaved best model ({best_name}) -> models/best_model.joblib")


if __name__ == "__main__":
    main()
