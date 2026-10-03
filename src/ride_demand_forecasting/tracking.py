"""Optional MLflow experiment tracking for training runs.

Turned on by setting `MLFLOW_TRACKING_URI` (see .env.example). Without it, training
skips tracking entirely, so the pipeline runs with no external service.
"""

import logging
import os
from pathlib import Path

import pandas as pd

from ride_demand_forecasting.config import LGBM_PARAMS, MLFLOW_EXPERIMENT

logger = logging.getLogger(__name__)


def run_params(artifact: dict) -> dict:
    """What defines a training run: the data it saw and how the model was configured."""
    eval_months = artifact["eval_train_months"]
    return {
        "train_first_month": artifact["train_months"][0],
        "train_last_month": artifact["train_months"][-1],
        "n_train_months": len(artifact["train_months"]),
        "eval_train_last_month": eval_months[-1] if eval_months else None,
        "test_month": artifact["test_month"],
        "window_months": artifact["window_months"],
        "gap_months": artifact["gap_months"],
        "n_pickup_zones": len(artifact["zones"]),
        **{f"lgbm_{name}": value for name, value in LGBM_PARAMS.items()},
    }


def log_training_run(artifact: dict, model_path: str | Path) -> str | None:
    """Log one training run and its model file to MLflow; returns the run id, or None
    if tracking is off."""
    if not os.environ.get("MLFLOW_TRACKING_URI"):
        logger.info("MLFLOW_TRACKING_URI is not set - skipping experiment tracking")
        return None

    # Imported here so that serving and untracked training never load MLflow.
    import mlflow

    mlflow.set_experiment(MLFLOW_EXPERIMENT)
    with mlflow.start_run(run_name=f"train-{artifact['train_months'][-1]}") as run:
        # `trained_at` is what lets the promotion step find this run again from the
        # model file alone.
        mlflow.set_tags(
            {"trained_at": artifact["trained_at"], "model_artifact": Path(model_path).name}
        )
        mlflow.log_params(run_params(artifact))
        mlflow.log_metrics(
            {
                **artifact["metrics"],
                "n_train_rows": artifact["n_train_rows"],
                "n_test_rows": artifact["n_test_rows"],
            }
        )
        mlflow.log_artifact(str(model_path))
    logger.info("Logged run %s to MLflow experiment %r", run.info.run_id, MLFLOW_EXPERIMENT)
    return run.info.run_id


def log_backtest(summary: pd.DataFrame, gap_months: int) -> list[str]:
    """Log one run per compared setting (window length and zone count) from a backtest
    summary; returns the run ids (empty if tracking is off)."""
    if not os.environ.get("MLFLOW_TRACKING_URI"):
        logger.info("MLFLOW_TRACKING_URI is not set - skipping experiment tracking")
        return []

    import mlflow

    mlflow.set_experiment(MLFLOW_EXPERIMENT)
    run_ids = []
    for row in summary.itertuples():
        run_name = f"backtest-window-{row.window_months}-zones-{row.n_zones}"
        with mlflow.start_run(run_name=run_name) as run:
            mlflow.log_params(
                {
                    "window_months": row.window_months,
                    "n_pickup_zones": row.n_zones,
                    "gap_months": gap_months,
                    "test_months": row.test_months,
                }
            )
            mlflow.log_metrics(
                {
                    "mae": row.mae,
                    "rmse": row.rmse,
                    "wape": row.wape,
                    "smallest_zone_share": row.smallest_zone_share,
                }
            )
        run_ids.append(run.info.run_id)
    logger.info("Logged %d backtest run(s) to MLflow", len(run_ids))
    return run_ids


def log_tuning(results: pd.DataFrame) -> str | None:
    """Log the best trial of a hyperparameter search, next to the score of the current
    configuration; returns the run id, or None if tracking is off."""
    if not os.environ.get("MLFLOW_TRACKING_URI"):
        logger.info("MLFLOW_TRACKING_URI is not set - skipping experiment tracking")
        return None

    import mlflow

    best = results.iloc[0]
    current = results[results["trial"] == "current"].iloc[0]
    mlflow.set_experiment(MLFLOW_EXPERIMENT)
    with mlflow.start_run(run_name="tune") as run:
        mlflow.log_params(
            {
                "best_trial": best["trial"],
                "n_trials": len(results) - 1,
                "folds": int(best["folds"]),
                **{f"lgbm_{name}": value for name, value in best["params"].items()},
            }
        )
        mlflow.log_metrics(
            {
                "wape": best["wape"],
                "mae": best["mae"],
                "rmse": best["rmse"],
                "current_wape": current["wape"],
            }
        )
    logger.info("Logged tuning run %s to MLflow", run.info.run_id)
    return run.info.run_id
