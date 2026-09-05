from pathlib import Path

from ride_demand_forecasting.config import N_PICKUP_ZONES
from ride_demand_forecasting.inference import PredictionService
from ride_demand_forecasting.train import train


def test_train_produces_a_complete_artifact(raw_trips_csv: Path) -> None:
    artifact = train(raw_data_path=raw_trips_csv)

    assert artifact["zones"] == list(range(N_PICKUP_ZONES))
    assert artifact["feature_order"] == [
        "pickup_zone",
        "hour",
        "avg_ride_duration_min",
        "avg_ride_distance",
    ]
    assert artifact["metrics"]["mae"] >= 0
    assert artifact["metrics"]["rmse"] >= 0
    assert artifact["n_train_rows"] > 0
    assert artifact["n_test_rows"] > 0
    assert len(artifact["zone_hour_profile"]) == N_PICKUP_ZONES * 24


def test_saved_artifact_round_trips_through_prediction_service(model_artifact_path: Path) -> None:
    service = PredictionService(model_path=model_artifact_path)

    prediction = service.predict_zone(pickup_zone=0, hour=12)

    assert isinstance(prediction, float)
    assert prediction >= 0
