"""Global + local explainability and calibration.

Exposes helpers used by both the offline report and the Streamlit app:
  - permutation_importance_df : model-agnostic global importance
  - shap_global / shap_local   : SHAP on the transformed feature space
  - calibration_data           : reliability curve points
  - top_risk_drivers           : per-prediction drivers for the API response
"""
import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.inspection import permutation_importance

from src import config as C
from src.features import feature_names


def _split(pipe):
    return pipe.named_steps["preprocessor"], pipe.named_steps["classifier"]


def permutation_importance_df(pipe, X, y, n_repeats=5, max_rows=2000):
    if len(X) > max_rows:
        X = X.sample(max_rows, random_state=C.RANDOM_SEED)
        y = y.loc[X.index]
    r = permutation_importance(pipe, X, y, n_repeats=n_repeats,
                               random_state=C.RANDOM_SEED, scoring="roc_auc", n_jobs=-1)
    # permutation is over raw input columns, not expanded features
    cols = X.columns
    return (pd.DataFrame({"feature": cols, "importance": r.importances_mean,
                          "std": r.importances_std})
            .sort_values("importance", ascending=False).reset_index(drop=True))


def _transform(pipe, X):
    pre, clf = _split(pipe)
    Xt = pre.transform(X)
    if hasattr(Xt, "toarray"):
        Xt = Xt.toarray()
    return Xt, feature_names(pre), clf


def shap_global(pipe, X_background, max_rows=300):
    """Return (feature_names, mean_abs_shap) for a tree/linear model."""
    import shap
    if len(X_background) > max_rows:
        X_background = X_background.sample(max_rows, random_state=C.RANDOM_SEED)
    Xt, names, clf = _transform(pipe, X_background)
    try:
        explainer = shap.TreeExplainer(clf)
        sv = explainer.shap_values(Xt)
    except Exception:
        explainer = shap.LinearExplainer(clf, Xt)
        sv = explainer.shap_values(Xt)
    if isinstance(sv, list):
        sv = sv[1] if len(sv) > 1 else sv[0]
    mean_abs = np.abs(sv).mean(axis=0)
    imp = (pd.DataFrame({"feature": names, "mean_abs_shap": mean_abs})
           .sort_values("mean_abs_shap", ascending=False).reset_index(drop=True))
    return imp


def top_risk_drivers(pipe, X_row: pd.DataFrame, k=5):
    """Per-prediction top-k drivers via SHAP; falls back to model coeffs/importances."""
    import shap
    Xt, names, clf = _transform(pipe, X_row)
    try:
        explainer = shap.TreeExplainer(clf)
        sv = explainer.shap_values(Xt)
        if isinstance(sv, list):
            sv = sv[1] if len(sv) > 1 else sv[0]
        contrib = sv[0]
    except Exception:
        if hasattr(clf, "coef_"):
            contrib = clf.coef_[0] * Xt[0]
        elif hasattr(clf, "feature_importances_"):
            contrib = clf.feature_importances_
        else:
            contrib = np.zeros(len(names))
    order = np.argsort(np.abs(contrib))[::-1][:k]
    return [{"feature": names[i], "contribution": float(contrib[i])} for i in order]


def calibration_data(pipe, X, y, n_bins=10):
    proba = pipe.predict_proba(X)[:, 1]
    frac_pos, mean_pred = calibration_curve(y, proba, n_bins=n_bins, strategy="quantile")
    return mean_pred, frac_pos


def lime_explanation(pipe, X_train, X_row, num_features=8):
    """LIME tabular explanation on the transformed feature space."""
    from lime.lime_tabular import LimeTabularExplainer
    pre, clf = _split(pipe)
    Xt_train = pre.transform(X_train.sample(min(2000, len(X_train)),
                                            random_state=C.RANDOM_SEED))
    if hasattr(Xt_train, "toarray"):
        Xt_train = Xt_train.toarray()
    names = feature_names(pre)
    explainer = LimeTabularExplainer(
        Xt_train, feature_names=names, class_names=["paid", "default"],
        discretize_continuous=True, mode="classification")
    Xt_row = pre.transform(X_row)
    if hasattr(Xt_row, "toarray"):
        Xt_row = Xt_row.toarray()
    exp = explainer.explain_instance(Xt_row[0], clf.predict_proba,
                                     num_features=num_features)
    return exp.as_list()
