import datetime as dt
import json
from pathlib import Path

import pandas as pd
import pytest

from ride_demand_forecasting.config import N_PICKUP_ZONES
from ride_demand_forecasting.inference import PredictionService
from ride_demand_forecasting.ingest import CENTROIDS_FILENAME
from ride_demand_forecasting.train import save_artifact, save_metrics, train, window_on_disk


def test_train_produces_a_complete_artifact(trained_artifact: dict) -> None:
    artifact = trained_artifact

    assert artifact["zones"] == list(range(N_PICKUP_ZONES))
    assert artifact["feature_order"] == [
        "pickup_zone",
        "hour",
        "day_of_week",
        "avg_ride_duration_min",
        "avg_ride_distance",
    ]
    assert artifact["metrics"]["mae"] >= 0
    assert artifact["metrics"]["rmse"] >= 0
    assert artifact["metrics"]["wape"] >= 0
    assert artifact["n_train_rows"] > 0
    assert artifact["n_test_rows"] > 0
    assert len(artifact["zone_hour_profile"]) == N_PICKUP_ZONES * 24
    assert not artifact["zone_hour_profile"].isna().any().any()


def test_train_maps_every_location_to_a_pickup_zone(
    trained_artifact: dict, tlc_data_dir: Path
) -> None:
    zone_map = trained_artifact["zone_map"]
    location_ids = pd.read_csv(tlc_data_dir / CENTROIDS_FILENAME)["location_id"]

    assert sorted(zone_map) == location_ids.tolist()
    assert set(zone_map.values()) == set(range(N_PICKUP_ZONES))


def test_train_evaluates_with_a_gap_before_the_test_month(trained_artifact: dict) -> None:
    # The saved model uses every month on disk; the evaluation model stops two months
    # before the month it is tested on.
    assert trained_artifact["train_months"] == [
        "2026-04",
        "2026-05",
        "2026-06",
        "2026-07",
        "2026-08",
    ]
    assert trained_artifact["eval_train_months"] == ["2026-04", "2026-05", "2026-06"]
    assert trained_artifact["test_month"] == "2026-08"


def test_train_honours_the_window_length(tlc_data_dir: Path) -> None:
    artifact = train(data_dir=tlc_data_dir, window_months=2, n_zones=5)

    assert artifact["train_months"] == ["2026-07", "2026-08"]
    assert artifact["eval_train_months"] == ["2026-05", "2026-06"]


def test_window_on_disk_never_reaches_past_the_calendar_window() -> None:
    available = [pd.Period(month, freq="M") for month in ("2026-03", "2026-05", "2026-08")]

    window = window_on_disk(pd.Period("2026-08", freq="M"), 4, available)

    # May to August: March is on disk but outside the window, June and July are missing.
    assert [str(month) for month in window] == ["2026-05", "2026-08"]
    assert window_on_disk(pd.Period("2026-01", freq="M"), 2, available) == []


def test_train_skips_evaluation_without_enough_history(tmp_path: Path, make_tlc_data_dir) -> None:
    data_dir = make_tlc_data_dir(tmp_path / "tlc", ["2026-08"])

    artifact = train(data_dir=data_dir, n_zones=5)

    assert artifact["train_months"] == ["2026-08"]
    assert artifact["metrics"] == {}
    assert artifact["test_month"] is None
    assert artifact["n_test_rows"] == 0


def test_train_fails_clearly_when_nothing_is_ingested(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="ride-demand-ingest"):
        train(data_dir=tmp_path)


def test_saved_artifact_round_trips_through_prediction_service(model_artifact_path: Path) -> None:
    service = PredictionService(model_path=model_artifact_path)

    prediction = service.predict_zone(pickup_zone=0, hour=12, date=dt.date(2026, 9, 14))

    assert isinstance(prediction, float)
    assert prediction >= 0


def test_predictions_follow_the_day_of_the_week(model_artifact_path: Path) -> None:
    service = PredictionService(model_path=model_artifact_path)
    monday, saturday = dt.date(2026, 9, 14), dt.date(2026, 9, 19)

    def total(date: dt.date) -> float:
        return sum(prediction for _, prediction in service.rank(hour=12, date=date))

    # The synthetic data has twice the trips on weekend days.
    assert total(saturday) > 1.5 * total(monday)


def test_age_days_is_near_zero_right_after_training(tmp_path: Path, trained_artifact: dict) -> None:
    artifact = {**trained_artifact, "trained_at": pd.Timestamp.now("UTC").isoformat()}
    model_path = tmp_path / "fresh_model.joblib"
    save_artifact(artifact, model_path=model_path)

    service = PredictionService(model_path=model_path)

    assert 0 <= service.age_days() < 0.01  # well under a minute in days


def test_age_days_reflects_an_older_artifact(tmp_path: Path, trained_artifact: dict) -> None:
    trained_at = (pd.Timestamp.now("UTC") - pd.Timedelta(days=45)).isoformat()
    model_path = tmp_path / "old_model.joblib"
    save_artifact({**trained_artifact, "trained_at": trained_at}, model_path=model_path)

    service = PredictionService(model_path=model_path)

    assert service.age_days() == pytest.approx(45, abs=0.01)


def test_save_metrics_writes_evaluation_metrics_and_row_counts(
    tmp_path: Path, trained_artifact: dict
) -> None:
    metrics_path = tmp_path / "out" / "metrics.json"

    save_metrics(trained_artifact, metrics_path=metrics_path)

    metrics = json.loads(metrics_path.read_text())
    assert metrics == {
        "mae": trained_artifact["metrics"]["mae"],
        "rmse": trained_artifact["metrics"]["rmse"],
        "wape": trained_artifact["metrics"]["wape"],
        "n_train_rows": trained_artifact["n_train_rows"],
        "n_test_rows": trained_artifact["n_test_rows"],
    }
