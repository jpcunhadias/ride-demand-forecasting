import datetime as dt
from zoneinfo import ZoneInfo

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from ride_demand_forecasting.api.main import app
from ride_demand_forecasting.config import N_PICKUP_ZONES
from ride_demand_forecasting.train import save_artifact


@pytest.fixture
def client(model_artifact_path, monkeypatch):
    monkeypatch.setenv("RDF_MODEL_PATH", str(model_artifact_path))
    monkeypatch.delenv("MLFLOW_TRACKING_URI", raising=False)
    with TestClient(app) as test_client:
        yield test_client


def test_health(client: TestClient) -> None:
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok", "model_source": "file", "model_version": None}


def test_predict_by_zone(client: TestClient) -> None:
    response = client.post("/predict", json={"pickup_zone": 0, "hour": 12})

    assert response.status_code == 200
    body = response.json()
    assert body["pickup_zone"] == 0
    assert body["hour"] == 12
    assert body["predicted_avg_daily_ride_count"] >= 0


def test_predict_defaults_to_today_in_new_york(client: TestClient) -> None:
    before = dt.datetime.now(ZoneInfo("America/New_York")).date()
    response = client.post("/predict", json={"pickup_zone": 0, "hour": 12})
    after = dt.datetime.now(ZoneInfo("America/New_York")).date()

    assert response.status_code == 200
    assert before.isoformat() <= response.json()["date"] <= after.isoformat()


def test_predict_uses_the_requested_date(client: TestClient) -> None:
    def predict(date: str) -> dict:
        response = client.post("/predict", json={"pickup_zone": 0, "hour": 12, "date": date})
        assert response.status_code == 200
        return response.json()

    monday, saturday = predict("2026-09-14"), predict("2026-09-19")

    assert monday["date"] == "2026-09-14"
    assert saturday["date"] == "2026-09-19"
    assert saturday["predicted_avg_daily_ride_count"] != monday["predicted_avg_daily_ride_count"]


def test_predict_rejects_an_invalid_date(client: TestClient) -> None:
    response = client.post("/predict", json={"pickup_zone": 0, "hour": 12, "date": "not-a-date"})

    assert response.status_code == 422


def test_predict_by_coords(client: TestClient) -> None:
    response = client.post("/predict", json={"lat": 40.75, "lng": -73.98, "hour": 8})

    assert response.status_code == 200
    body = response.json()
    assert 0 <= body["pickup_zone"] < N_PICKUP_ZONES


def test_predict_rejects_both_zone_and_coords(client: TestClient) -> None:
    response = client.post(
        "/predict", json={"pickup_zone": 0, "lat": 40.75, "lng": -73.98, "hour": 8}
    )

    assert response.status_code == 422


def test_predict_rejects_neither_zone_nor_coords(client: TestClient) -> None:
    response = client.post("/predict", json={"hour": 8})

    assert response.status_code == 422


def test_predict_rejects_out_of_bounds_coords(client: TestClient) -> None:
    response = client.post("/predict", json={"lat": 0.0, "lng": 0.0, "hour": 8})

    assert response.status_code == 422


def test_predict_rejects_invalid_hour(client: TestClient) -> None:
    response = client.post("/predict", json={"pickup_zone": 0, "hour": 24})

    assert response.status_code == 422


def test_predict_rejects_unknown_zone(client: TestClient) -> None:
    response = client.post("/predict", json={"pickup_zone": 999, "hour": 8})

    assert response.status_code == 422


def test_fresh_model_does_not_log_staleness_warning(client: TestClient, caplog) -> None:
    with caplog.at_level("WARNING"):
        client.get("/health")

    assert not any("consider retraining" in r.message for r in caplog.records)


def test_stale_model_logs_staleness_warning(
    tmp_path, trained_artifact, monkeypatch, caplog
) -> None:
    trained_at = (pd.Timestamp.now("UTC") - pd.Timedelta(days=45)).isoformat()
    model_path = tmp_path / "stale_model.joblib"
    save_artifact({**trained_artifact, "trained_at": trained_at}, model_path=model_path)
    monkeypatch.setenv("RDF_MODEL_PATH", str(model_path))

    with caplog.at_level("WARNING"), TestClient(app):
        pass

    assert any("consider retraining" in r.message for r in caplog.records)


def test_rankings(client: TestClient) -> None:
    response = client.get("/rankings", params={"hour": 18, "top_n": 5, "date": "2026-09-19"})

    assert response.status_code == 200
    body = response.json()
    assert body["hour"] == 18
    assert body["date"] == "2026-09-19"
    assert len(body["rankings"]) == 5
    predictions = [r["predicted_avg_daily_ride_count"] for r in body["rankings"]]
    assert predictions == sorted(predictions, reverse=True)
