import json
import time
from pathlib import Path

import joblib
import mlflow
import pandas as pd
import pytest
from fastapi.testclient import TestClient
from mlflow import MlflowClient
from mlflow.exceptions import MlflowException

from ride_demand_forecasting import registry
from ride_demand_forecasting.api import main as api_main
from ride_demand_forecasting.api.main import app
from ride_demand_forecasting.ingest import CENTROIDS_FILENAME
from ride_demand_forecasting.promote import (
    ALREADY_CURRENT,
    decide,
    describe_versions,
    promote,
    record_decision,
    score_artifact,
)
from ride_demand_forecasting.tracking import log_training_run
from ride_demand_forecasting.train import load_month, save_artifact, train

# Tolerances that make the outcome certain whatever the two scores are, so these
# tests check the flow; the rule's arithmetic is checked on `decide` directly.
ALWAYS_WITHIN, NEVER_WITHIN = 100.0, -0.99


@pytest.fixture(scope="module")
def july_model(tlc_data_dir: Path) -> dict:
    return train(data_dir=tlc_data_dir, end_month=pd.Period("2026-07", freq="M"), n_zones=5)


@pytest.fixture(scope="module")
def august_model(tlc_data_dir: Path) -> dict:
    return train(data_dir=tlc_data_dir, end_month=pd.Period("2026-08", freq="M"), n_zones=5)


def log_model(artifact: dict, directory: Path, name: str) -> Path:
    """Save a model file and log its training run, as `ride-demand-train` does."""
    path = directory / name / "model.joblib"
    save_artifact(artifact, model_path=path)
    log_training_run(artifact, path)
    return path


def retrained(artifact: dict) -> dict:
    """The same model as if it had been trained again a moment later."""
    return {**artifact, "trained_at": pd.Timestamp.now("UTC").isoformat()}


@pytest.mark.parametrize(
    ("candidate", "current", "promoted"),
    [
        (0.25, None, True),  # nothing in service yet
        (None, 0.25, False),  # the new model was never evaluated
        (None, None, False),
        (0.24, 0.25, True),  # better
        (0.26, 0.25, True),  # 4% worse: inside the 5% margin
        (0.2625, 0.25, True),  # exactly on the limit
        (0.27, 0.25, False),  # 8% worse
    ],
)
def test_decide_promotes_unless_clearly_worse(
    candidate: float | None, current: float | None, promoted: bool
) -> None:
    assert decide(candidate, current, tolerance=0.05)[0] is promoted


def test_promote_does_nothing_without_mlflow(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, august_model: dict, tlc_data_dir: Path
) -> None:
    monkeypatch.delenv("MLFLOW_TRACKING_URI", raising=False)
    path = tmp_path / "model.joblib"
    save_artifact(august_model, model_path=path)

    decision = promote(model_path=path, data_dir=tlc_data_dir)

    assert decision["promoted"] is False
    assert "MLflow is not configured" in decision["reason"]


def test_promote_refuses_a_model_that_was_never_logged(
    tmp_path: Path, mlflow_store: str, august_model: dict, tlc_data_dir: Path
) -> None:
    path = tmp_path / "model.joblib"
    save_artifact(august_model, model_path=path)

    with pytest.raises(RuntimeError, match="was not logged to MLflow"):
        promote(model_path=path, data_dir=tlc_data_dir)


def test_first_model_is_promoted(
    tmp_path: Path, mlflow_store: str, july_model: dict, tlc_data_dir: Path
) -> None:
    decision = promote(model_path=log_model(july_model, tmp_path, "july"), data_dir=tlc_data_dir)

    assert decision["promoted"] is True
    assert decision["reason"] == "there is no current model"
    assert (decision["candidate_version"], decision["previous_version"]) == (1, None)
    assert int(registry.champion().version) == 1


def test_newer_model_is_compared_on_the_same_month_with_the_same_data(
    tmp_path: Path, mlflow_store: str, july_model: dict, august_model: dict, tlc_data_dir: Path
) -> None:
    promote(model_path=log_model(july_model, tmp_path, "july"), data_dir=tlc_data_dir)

    decision = promote(
        model_path=log_model(august_model, tmp_path, "august"), data_dir=tlc_data_dir
    )

    assert decision["promoted"] is True
    assert (decision["candidate_version"], decision["previous_version"]) == (2, 1)
    assert decision["test_month"] == "2026-08"
    # The July model, as deployed, on August - a month it never saw.
    centroids = pd.read_csv(tlc_data_dir / CENTROIDS_FILENAME)
    august = load_month(pd.Period("2026-08", freq="M"), tlc_data_dir, centroids)
    assert decision["current_wape"] == pytest.approx(
        score_artifact(july_model, august, centroids)["wape"]
    )
    # The new model's stand-in is trained to July as well. Nothing in the pipeline
    # changed between the two, so with the same data they score the same - and not
    # the worse figure of the ordinary evaluation copy, which stops at June.
    assert decision["candidate_wape"] == pytest.approx(decision["current_wape"])
    assert decision["evaluation_wape"] == august_model["metrics"]["wape"]
    assert int(registry.champion().version) == 2


def test_clearly_worse_model_is_registered_but_not_promoted(
    tmp_path: Path, mlflow_store: str, july_model: dict, august_model: dict, tlc_data_dir: Path
) -> None:
    promote(model_path=log_model(july_model, tmp_path, "july"), data_dir=tlc_data_dir)

    decision = promote(
        model_path=log_model(august_model, tmp_path, "august"),
        data_dir=tlc_data_dir,
        tolerance=NEVER_WITHIN,
    )

    assert decision["promoted"] is False
    assert "above the limit" in decision["reason"]
    assert decision["candidate_version"] == 2
    assert int(registry.champion().version) == 1


def test_unevaluated_model_is_never_promoted(
    tmp_path: Path, mlflow_store: str, august_model: dict, tlc_data_dir: Path
) -> None:
    unevaluated = {**august_model, "metrics": {}, "test_month": None, "eval_train_months": []}

    decision = promote(
        model_path=log_model(unevaluated, tmp_path, "unevaluated"), data_dir=tlc_data_dir
    )

    # Even with nothing in service, which would otherwise promote the first model.
    assert decision["promoted"] is False
    assert "was not evaluated" in decision["reason"]
    assert decision["candidate_version"] == 1
    assert registry.champion() is None


def test_promoting_the_same_model_again_changes_nothing(
    tmp_path: Path, mlflow_store: str, august_model: dict, tlc_data_dir: Path
) -> None:
    path = log_model(august_model, tmp_path, "august")
    promote(model_path=path, data_dir=tlc_data_dir)

    decision = promote(model_path=path, data_dir=tlc_data_dir)

    assert decision["promoted"] is False
    assert decision["reason"] == "this model is already the current one"
    assert [int(v.version) for v in registry.list_versions()] == [1]
    # The record of the decision that promoted it is left as it was.
    assert describe_versions()["promoted"].tolist() == ["true"]


def test_retrain_on_the_same_data_is_compared_evaluation_to_evaluation(
    tmp_path: Path, mlflow_store: str, august_model: dict, tlc_data_dir: Path
) -> None:
    promote(model_path=log_model(august_model, tmp_path, "first"), data_dir=tlc_data_dir)

    decision = promote(
        model_path=log_model(retrained(august_model), tmp_path, "second"), data_dir=tlc_data_dir
    )

    # The current model has seen August, so it can't be re-scored on it; its own
    # evaluation was on August under the same conditions, and is used instead.
    assert decision["current_wape"] == august_model["metrics"]["wape"]
    assert decision["promoted"] is True
    assert int(registry.champion().version) == 2


def test_model_older_than_the_current_one_is_not_promoted(
    tmp_path: Path, mlflow_store: str, july_model: dict, august_model: dict, tlc_data_dir: Path
) -> None:
    promote(model_path=log_model(august_model, tmp_path, "august"), data_dir=tlc_data_dir)

    decision = promote(
        model_path=log_model(july_model, tmp_path, "july"),
        data_dir=tlc_data_dir,
        tolerance=ALWAYS_WITHIN,
    )

    assert decision["promoted"] is False
    assert "no fair comparison" in decision["reason"]
    assert int(registry.champion().version) == 1


def test_rollback_points_the_service_at_an_earlier_version(
    tmp_path: Path, mlflow_store: str, july_model: dict, august_model: dict, tlc_data_dir: Path
) -> None:
    promote(model_path=log_model(july_model, tmp_path, "july"), data_dir=tlc_data_dir)
    promote(
        model_path=log_model(august_model, tmp_path, "august"),
        data_dir=tlc_data_dir,
        tolerance=ALWAYS_WITHIN,
    )

    registry.set_champion(1)

    assert int(registry.champion().version) == 1
    versions = describe_versions()
    assert versions["version"].tolist() == [2, 1]
    assert versions["current"].tolist() == ["", "*"]
    assert versions["trained_to"].tolist() == ["2026-08", "2026-07"]
    with pytest.raises(Exception, match="not found|does not exist|RESOURCE_DOES_NOT_EXIST"):
        registry.set_champion(99)
    assert int(registry.champion().version) == 1


def test_api_serves_the_promoted_model(
    tmp_path: Path,
    mlflow_store: str,
    monkeypatch: pytest.MonkeyPatch,
    july_model: dict,
    model_artifact_path: Path,
    tlc_data_dir: Path,
) -> None:
    promote(model_path=log_model(july_model, tmp_path, "july"), data_dir=tlc_data_dir)
    # The file fallback is a different model, with 20 zones to the promoted one's 5.
    monkeypatch.setenv("RDF_MODEL_PATH", str(model_artifact_path))

    with TestClient(app) as client:
        health = client.get("/health").json()
        zones = client.get("/rankings", params={"hour": 12}).json()["rankings"]

    assert health == {"status": "ok", "model_source": "registry", "model_version": "1"}
    assert len(zones) == 5


def test_api_falls_back_to_the_model_file_when_nothing_is_promoted(
    mlflow_store: str, monkeypatch: pytest.MonkeyPatch, model_artifact_path: Path
) -> None:
    monkeypatch.setenv("RDF_MODEL_PATH", str(model_artifact_path))

    with TestClient(app) as client:
        health = client.get("/health").json()

    assert health == {"status": "ok", "model_source": "file", "model_version": None}


def test_api_falls_back_to_the_model_file_when_mlflow_is_unreachable(
    monkeypatch: pytest.MonkeyPatch, model_artifact_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    monkeypatch.setenv("MLFLOW_TRACKING_URI", "http://127.0.0.1:9")
    monkeypatch.setenv("MLFLOW_HTTP_REQUEST_MAX_RETRIES", "0")
    monkeypatch.setenv("MLFLOW_HTTP_REQUEST_TIMEOUT", "2")
    monkeypatch.setenv("RDF_MODEL_PATH", str(model_artifact_path))

    with caplog.at_level("ERROR"), TestClient(app) as client:
        health = client.get("/health").json()

    assert health["model_source"] == "file"
    # It really tried the server and failed, rather than finding an empty registry.
    assert any("Could not load the promoted model" in r.message for r in caplog.records)


def test_api_falls_back_when_the_promoted_model_has_bad_metadata(
    tmp_path: Path,
    mlflow_store: str,
    monkeypatch: pytest.MonkeyPatch,
    july_model: dict,
    model_artifact_path: Path,
    tlc_data_dir: Path,
) -> None:
    broken = {**july_model, "trained_at": "not a timestamp"}
    promote(model_path=log_model(broken, tmp_path, "broken"), data_dir=tlc_data_dir)
    assert registry.champion() is not None
    monkeypatch.setenv("RDF_MODEL_PATH", str(model_artifact_path))

    with TestClient(app) as client:
        health = client.get("/health").json()

    assert health["model_source"] == "file"


def test_api_gives_up_on_a_registry_that_does_not_answer_in_time(
    mlflow_store: str, monkeypatch: pytest.MonkeyPatch, model_artifact_path: Path
) -> None:
    monkeypatch.setenv("RDF_MODEL_PATH", str(model_artifact_path))
    monkeypatch.setattr(api_main, "REGISTRY_LOAD_TIMEOUT_S", 0.2)
    monkeypatch.setattr(registry, "champion", lambda: time.sleep(5))

    started = time.perf_counter()
    with TestClient(app) as client:
        health = client.get("/health").json()
        elapsed = time.perf_counter() - started

    assert health["model_source"] == "file"
    assert elapsed < 4


def test_registry_error_is_not_mistaken_for_an_empty_registry(
    tmp_path: Path,
    mlflow_store: str,
    monkeypatch: pytest.MonkeyPatch,
    july_model: dict,
    august_model: dict,
    tlc_data_dir: Path,
) -> None:
    promote(model_path=log_model(july_model, tmp_path, "july"), data_dir=tlc_data_dir)
    august_path = log_model(august_model, tmp_path, "august")
    lookup = MlflowClient.get_model_version_by_alias

    def failing_lookup(self, name, alias):
        raise MlflowException("server error", error_code=1)  # INTERNAL_ERROR

    monkeypatch.setattr(MlflowClient, "get_model_version_by_alias", failing_lookup)
    with pytest.raises(MlflowException, match="server error"):
        promote(model_path=august_path, data_dir=tlc_data_dir, tolerance=ALWAYS_WITHIN)

    # Nothing was promoted on the strength of a failed lookup.
    monkeypatch.setattr(MlflowClient, "get_model_version_by_alias", lookup)
    assert int(registry.champion().version) == 1


def test_run_whose_model_upload_failed_cannot_be_promoted(
    tmp_path: Path,
    mlflow_store: str,
    monkeypatch: pytest.MonkeyPatch,
    august_model: dict,
    tlc_data_dir: Path,
) -> None:
    path = tmp_path / "model.joblib"
    save_artifact(august_model, model_path=path)

    def failing_upload(*args, **kwargs):
        raise OSError("upload failed")

    with monkeypatch.context() as patch:
        patch.setattr(mlflow, "log_artifact", failing_upload)
        with pytest.raises(OSError, match="upload failed"):
            log_training_run(august_model, path)

    with pytest.raises(RuntimeError, match="was not logged to MLflow"):
        promote(model_path=path, data_dir=tlc_data_dir)
    assert registry.list_versions() == []


def test_model_file_that_differs_from_the_logged_one_is_refused(
    tmp_path: Path, mlflow_store: str, august_model: dict, tlc_data_dir: Path
) -> None:
    path = log_model(august_model, tmp_path, "august")
    # Same training timestamp, different contents.
    joblib.dump({**august_model, "metrics": {"wape": 0.0, "mae": 0.0, "rmse": 0.0}}, path)

    with pytest.raises(RuntimeError, match="is not the file its training run logged"):
        promote(model_path=path, data_dir=tlc_data_dir)
    assert registry.list_versions() == []


def test_stored_scores_are_only_compared_when_evaluated_the_same_way(
    tmp_path: Path, mlflow_store: str, august_model: dict, tlc_data_dir: Path
) -> None:
    promote(model_path=log_model(august_model, tmp_path, "first"), data_dir=tlc_data_dir)
    # Evaluated on the same month, but with training data reaching a month further.
    closer = {**retrained(august_model), "eval_train_months": ["2026-06", "2026-07"]}

    decision = promote(
        model_path=log_model(closer, tmp_path, "closer"),
        data_dir=tlc_data_dir,
        tolerance=ALWAYS_WITHIN,
    )

    assert decision["promoted"] is False
    assert "no fair comparison" in decision["reason"]
    assert int(registry.champion().version) == 1


def test_no_comparison_when_the_matching_month_is_not_on_disk(
    tmp_path: Path,
    mlflow_store: str,
    make_tlc_data_dir,
    july_model: dict,
    august_model: dict,
    tlc_data_dir: Path,
) -> None:
    promote(model_path=log_model(july_model, tmp_path, "july"), data_dir=tlc_data_dir)
    without_july = make_tlc_data_dir(tmp_path / "gap", ["2026-05", "2026-06", "2026-08"])

    decision = promote(
        model_path=log_model(august_model, tmp_path, "august"),
        data_dir=without_july,
        tolerance=ALWAYS_WITHIN,
    )

    # A copy trained only to June would not be on equal terms with a model trained
    # to July.
    assert decision["promoted"] is False
    assert "the data for 2026-07 is not on disk" in decision["reason"]


def test_unloadable_current_model_is_a_recorded_rejection(
    tmp_path: Path,
    mlflow_store: str,
    monkeypatch: pytest.MonkeyPatch,
    july_model: dict,
    august_model: dict,
    tlc_data_dir: Path,
) -> None:
    promote(model_path=log_model(july_model, tmp_path, "july"), data_dir=tlc_data_dir)
    august_path = log_model(august_model, tmp_path, "august")

    def missing(*args, **kwargs):
        raise FileNotFoundError("artifact is gone")

    monkeypatch.setattr(registry, "download_model", missing)
    decision = promote(model_path=august_path, data_dir=tlc_data_dir, tolerance=ALWAYS_WITHIN)

    assert decision["promoted"] is False
    assert "could not be loaded" in decision["reason"]
    assert describe_versions()["promoted"].tolist() == ["false", "true"]
    assert int(registry.champion().version) == 1


def test_decision_is_recorded_in_full_on_the_training_run(
    tmp_path: Path, mlflow_store: str, july_model: dict, august_model: dict, tlc_data_dir: Path
) -> None:
    promote(model_path=log_model(july_model, tmp_path, "july"), data_dir=tlc_data_dir)

    decision = promote(
        model_path=log_model(august_model, tmp_path, "august"), data_dir=tlc_data_dir
    )

    run = MlflowClient().get_run(registry.champion().run_id)
    assert json.loads(run.data.tags["promotion_decision"]) == decision
    assert run.data.metrics["promotion_candidate_wape"] == pytest.approx(decision["candidate_wape"])
    assert run.data.metrics["promotion_current_wape"] == pytest.approx(decision["current_wape"])
    assert {"candidate_version", "previous_version", "test_month", "evaluation_wape"} <= set(
        decision
    )


def test_failed_alias_move_is_finished_by_the_next_run(
    tmp_path: Path,
    mlflow_store: str,
    monkeypatch: pytest.MonkeyPatch,
    august_model: dict,
    tlc_data_dir: Path,
) -> None:
    path = log_model(august_model, tmp_path, "august")

    def failing_move(version):
        raise ConnectionError("lost the server")

    with monkeypatch.context() as patch:
        patch.setattr(registry, "set_champion", failing_move)
        with pytest.raises(ConnectionError):
            promote(model_path=path, data_dir=tlc_data_dir)
    assert registry.champion() is None

    decision = promote(model_path=path, data_dir=tlc_data_dir)

    assert decision["promoted"] is True
    assert int(registry.champion().version) == 1


def test_decision_file_survives_running_promotion_again(tmp_path: Path) -> None:
    path = tmp_path / "promotion.json"
    promoted = {"promoted": True, "reason": "there is no current model"}
    repeat = {"promoted": False, "reason": ALREADY_CURRENT}

    assert record_decision(promoted, path) is True
    assert record_decision(repeat, path) is False
    assert json.loads(path.read_text()) == promoted
    # With no earlier record there is nothing to protect, and the pipeline needs the file.
    assert record_decision(repeat, tmp_path / "fresh.json") is True


def test_values_with_quotes_are_refused_in_registry_searches(mlflow_store: str) -> None:
    with pytest.raises(ValueError, match="contains a quote"):
        registry.find_training_run("2026-08' or tags.trained_at != '")
