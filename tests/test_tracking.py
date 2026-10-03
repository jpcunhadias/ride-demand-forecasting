from pathlib import Path

import mlflow
import pandas as pd
import pytest

from ride_demand_forecasting.config import LGBM_PARAMS, MLFLOW_EXPERIMENT, N_PICKUP_ZONES
from ride_demand_forecasting.tracking import log_backtest, log_training_run, run_params


def test_log_training_run_is_skipped_without_a_tracking_uri(
    monkeypatch: pytest.MonkeyPatch, trained_artifact: dict, model_artifact_path: Path
) -> None:
    monkeypatch.delenv("MLFLOW_TRACKING_URI", raising=False)

    assert log_training_run(trained_artifact, model_artifact_path) is None


def test_run_params_describe_the_data_and_the_model(trained_artifact: dict) -> None:
    params = run_params(trained_artifact)

    assert params["train_first_month"] == "2026-04"
    assert params["train_last_month"] == "2026-08"
    assert params["n_train_months"] == 5
    assert params["eval_train_last_month"] == "2026-06"
    assert params["test_month"] == "2026-08"
    assert params["n_pickup_zones"] == N_PICKUP_ZONES
    assert params["lgbm_num_leaves"] == LGBM_PARAMS["num_leaves"]


def test_log_training_run_records_params_metrics_and_the_model_file(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    trained_artifact: dict,
    model_artifact_path: Path,
) -> None:
    # A local file store stands in for the tracking server. MLflow only allows it on
    # request, since real deployments are expected to use a server or a database.
    tracking_uri = (tmp_path / "mlruns").as_uri()
    monkeypatch.setenv("MLFLOW_ALLOW_FILE_STORE", "true")
    monkeypatch.setenv("MLFLOW_TRACKING_URI", tracking_uri)
    mlflow.set_tracking_uri(tracking_uri)

    run_id = log_training_run(trained_artifact, model_artifact_path)

    assert run_id is not None
    run = mlflow.get_run(run_id)
    assert mlflow.get_experiment(run.info.experiment_id).name == MLFLOW_EXPERIMENT
    assert run.data.params["train_last_month"] == "2026-08"
    assert run.data.params["window_months"] == "12"
    assert run.data.metrics["mae"] == pytest.approx(trained_artifact["metrics"]["mae"])
    assert run.data.metrics["n_train_rows"] == trained_artifact["n_train_rows"]
    logged = [f.path for f in mlflow.artifacts.list_artifacts(run_id=run_id)]
    assert logged == [model_artifact_path.name]


def test_log_backtest_records_one_run_per_window(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    tracking_uri = (tmp_path / "mlruns").as_uri()
    monkeypatch.setenv("MLFLOW_ALLOW_FILE_STORE", "true")
    monkeypatch.setenv("MLFLOW_TRACKING_URI", tracking_uri)
    mlflow.set_tracking_uri(tracking_uri)
    summary = pd.DataFrame(
        {
            "window_months": [3, 12],
            "test_months": [6, 6],
            "mae": [5.0, 4.0],
            "rmse": [8.0, 7.0],
            "wape": [0.30, 0.25],
        }
    )

    run_ids = log_backtest(summary, gap_months=2)

    assert len(run_ids) == 2
    run = mlflow.get_run(run_ids[1])
    assert run.info.run_name == "backtest-window-12"
    assert run.data.params == {"window_months": "12", "gap_months": "2", "test_months": "6"}
    assert run.data.metrics["wape"] == pytest.approx(0.25)


def test_log_backtest_is_skipped_without_a_tracking_uri(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("MLFLOW_TRACKING_URI", raising=False)

    assert log_backtest(pd.DataFrame({"window_months": [3]}), gap_months=2) == []
