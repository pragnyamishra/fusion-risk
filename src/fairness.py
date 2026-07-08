"""Fairness audit across income groups using Fairlearn.

Measures disparate impact and selection-rate / error-rate disparities in the
spirit of ECOA / Regulation B, which prohibit lending discrimination on
protected bases. Income group here is a demonstrative sensitive attribute;
in production you would audit legally protected classes where permitted.
"""
import numpy as np
import pandas as pd

from src import config as C


def audit(pipe, X, y, sensitive: pd.Series, threshold=0.5):
    """Return a per-group fairness table plus summary disparity metrics."""
    from fairlearn.metrics import (MetricFrame, selection_rate,
                                   false_negative_rate, false_positive_rate)
    from sklearn.metrics import accuracy_score, recall_score

    proba = pipe.predict_proba(X)[:, 1]
    pred = (proba >= threshold).astype(int)

    metrics = {
        "accuracy": accuracy_score,
        "recall": lambda yt, yp: recall_score(yt, yp, zero_division=0),
        "selection_rate": selection_rate,
        "false_negative_rate": false_negative_rate,
        "false_positive_rate": false_positive_rate,
    }
    mf = MetricFrame(metrics=metrics, y_true=y, y_pred=pred,
                     sensitive_features=sensitive)

    by_group = mf.by_group.copy()
    by_group["group_size"] = sensitive.value_counts().reindex(by_group.index).values

    sel = by_group["selection_rate"]
    # Disparate impact ratio (four-fifths rule): min/max selection rate
    di_ratio = float(sel.min() / sel.max()) if sel.max() > 0 else np.nan

    summary = {
        "disparate_impact_ratio": di_ratio,
        "passes_four_fifths_rule": bool(di_ratio >= 0.8) if not np.isnan(di_ratio) else None,
        "selection_rate_difference": float(mf.difference()["selection_rate"]),
        "false_negative_rate_difference": float(mf.difference()["false_negative_rate"]),
        "recall_difference": float(mf.difference()["recall"]),
    }
    return by_group.reset_index().rename(columns={"index": C.SENSITIVE_COL}), summary


if __name__ == "__main__":
    import joblib
    from src.data import build_subset, clean_and_engineer, temporal_split

    pipe = joblib.load(C.MODELS_DIR / "best_model.joblib")
    df = clean_and_engineer(build_subset())
    _, test_df, _ = temporal_split(df)
    table, summ = audit(pipe, test_df.drop(columns=[C.TARGET]),
                        test_df[C.TARGET], test_df[C.SENSITIVE_COL])
    print(table.to_string(index=False))
    print(summ)
