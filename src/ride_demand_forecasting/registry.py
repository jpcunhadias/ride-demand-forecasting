"""The MLflow model registry: which trained model is the one in service.

Every training run logged to MLflow can be registered as a version of one registered
model. The version carrying the `champion` alias is the model the API serves; moving
the alias is what promotion and rollback are.

MLflow is imported inside each function, so importing this module costs nothing when
tracking is off.
"""

import logging
import os
from pathlib import Path
from typing import Any

from ride_demand_forecasting.config import MLFLOW_EXPERIMENT, MLFLOW_MODEL_NAME

logger = logging.getLogger(__name__)

CHAMPION_ALIAS = "champion"


def tracking_enabled() -> bool:
    return bool(os.environ.get("MLFLOW_TRACKING_URI"))


def find_training_run(trained_at: str) -> Any | None:
    """The MLflow run that logged the model trained at `trained_at`, if there is one."""
    import mlflow

    runs = mlflow.search_runs(
        experiment_names=[MLFLOW_EXPERIMENT],
        filter_string=f"tags.trained_at = '{trained_at}'",
        output_format="list",
    )
    return runs[0] if runs else None


def register_run(run_id: str) -> Any:
    """The model version for a training run, creating it on first use."""
    from mlflow import MlflowClient
    from mlflow.exceptions import MlflowException

    client = MlflowClient()
    try:
        client.create_registered_model(MLFLOW_MODEL_NAME)
    except MlflowException:
        pass  # already there
    existing = client.search_model_versions(f"name = '{MLFLOW_MODEL_NAME}' and run_id = '{run_id}'")
    if existing:
        return existing[0]
    artifact = client.get_run(run_id).data.tags["model_artifact"]
    return client.create_model_version(
        MLFLOW_MODEL_NAME, source=f"runs:/{run_id}/{artifact}", run_id=run_id
    )


def champion() -> Any | None:
    """The model version currently in service, or None if nothing has been promoted."""
    from mlflow import MlflowClient
    from mlflow.exceptions import MlflowException

    try:
        return MlflowClient().get_model_version_by_alias(MLFLOW_MODEL_NAME, CHAMPION_ALIAS)
    except MlflowException:
        return None


def set_champion(version: str | int) -> None:
    from mlflow import MlflowClient

    client = MlflowClient()
    client.get_model_version(MLFLOW_MODEL_NAME, str(version))  # fails if it doesn't exist
    client.set_registered_model_alias(MLFLOW_MODEL_NAME, CHAMPION_ALIAS, str(version))


def download_model(version: Any, dest_dir: str | Path) -> Path:
    """Fetch a model version's file into `dest_dir` and return its path."""
    import mlflow
    from mlflow import MlflowClient

    artifact = MlflowClient().get_run(version.run_id).data.tags["model_artifact"]
    return Path(
        mlflow.artifacts.download_artifacts(
            run_id=version.run_id, artifact_path=artifact, dst_path=str(dest_dir)
        )
    )


def list_versions() -> list[Any]:
    """Every registered version, newest first."""
    from mlflow import MlflowClient

    versions = MlflowClient().search_model_versions(f"name = '{MLFLOW_MODEL_NAME}'")
    return sorted(versions, key=lambda version: int(version.version), reverse=True)
