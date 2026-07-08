"""Read MLflow-tracked runs from the local SQLite store for in-app display.

Used by the Streamlit "Experiments" tab so the experiment tracking is visible on the
deployed link without needing to run the separate `mlflow ui` server.

Primary path uses the MLflow client. If MLflow is unavailable or the store cannot be
opened, falls back to a direct read of the SQLite tables so the tab still renders.
"""
import sqlite3

import pandas as pd

from src import config as C

_DB_PATH = C.MLRUNS_DIR / "mlflow.db"
_METRIC_COLS = ["roc_auc", "pr_auc", "f1", "precision", "recall"]


def _db_exists() -> bool:
    return _DB_PATH.exists()


def _read_via_client() -> pd.DataFrame:
    import mlflow
    from mlflow.tracking import MlflowClient

    mlflow.set_tracking_uri(f"sqlite:///{_DB_PATH}")
    client = MlflowClient()
    exp = client.get_experiment_by_name(C.MLFLOW_EXPERIMENT)
    if exp is None:
        return pd.DataFrame()

    rows = []
    for run in client.search_runs([exp.experiment_id]):
        name = run.data.tags.get("mlflow.runName") or run.info.run_name or run.info.run_id[:8]
        row = {"run": name}
        for m in _METRIC_COLS:
            row[m] = run.data.metrics.get(m)
        row["_params"] = dict(run.data.params)
        rows.append(row)
    return pd.DataFrame(rows)


def _read_via_sqlite() -> pd.DataFrame:
    con = sqlite3.connect(str(_DB_PATH))
    try:
        exp = con.execute(
            "SELECT experiment_id FROM experiments WHERE name = ?",
            (C.MLFLOW_EXPERIMENT,),
        ).fetchone()
        if not exp:
            return pd.DataFrame()
        exp_id = exp[0]
        runs = con.execute(
            "SELECT run_uuid, name FROM runs WHERE experiment_id = ? "
            "AND lifecycle_stage = 'active'",
            (exp_id,),
        ).fetchall()

        rows = []
        for run_uuid, name in runs:
            row = {"run": name or run_uuid[:8]}
            metrics = dict(con.execute(
                "SELECT key, value FROM latest_metrics WHERE run_uuid = ?",
                (run_uuid,),
            ).fetchall())
            if not metrics:  # older stores may not have latest_metrics populated
                metrics = dict(con.execute(
                    "SELECT key, value FROM metrics WHERE run_uuid = ?",
                    (run_uuid,),
                ).fetchall())
            for m in _METRIC_COLS:
                row[m] = metrics.get(m)
            params = dict(con.execute(
                "SELECT key, value FROM params WHERE run_uuid = ?",
                (run_uuid,),
            ).fetchall())
            row["_params"] = params
            rows.append(row)
        return pd.DataFrame(rows)
    finally:
        con.close()


def load_runs() -> pd.DataFrame:
    """Return a DataFrame of tracked runs, one row per model run.

    Columns: run, roc_auc, pr_auc, f1, precision, recall, _params (dict).
    Empty DataFrame if no store is available.
    """
    if not _db_exists():
        return pd.DataFrame()
    try:
        df = _read_via_client()
        if not df.empty:
            return df.sort_values("roc_auc", ascending=False).reset_index(drop=True)
    except Exception:
        pass
    try:
        df = _read_via_sqlite()
        if not df.empty:
            return df.sort_values("roc_auc", ascending=False).reset_index(drop=True)
    except Exception:
        pass
    return pd.DataFrame()


if __name__ == "__main__":
    runs = load_runs()
    if runs.empty:
        print("No tracked runs found.")
    else:
        print(runs.drop(columns=["_params"]).to_string(index=False))
        print("\nTuned-run params sample:")
        tuned = runs[runs["run"].str.contains("Optuna", case=False, na=False)]
        if not tuned.empty:
            print(tuned.iloc[0]["_params"])