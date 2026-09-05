import pytest
from fastapi.testclient import TestClient

from ride_demand_forecasting.api.main import app
from ride_demand_forecasting.config import N_PICKUP_ZONES


@pytest.fixture
def client(model_artifact_path, monkeypatch):
    monkeypatch.setenv("RDF_MODEL_PATH", str(model_artifact_path))
    with TestClient(app) as test_client:
        yield test_client


def test_health(client: TestClient) -> None:
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_predict_by_zone(client: TestClient) -> None:
    response = client.post("/predict", json={"pickup_zone": 0, "hour": 12})

    assert response.status_code == 200
    body = response.json()
    assert body["pickup_zone"] == 0
    assert body["hour"] == 12
    assert body["predicted_avg_daily_ride_count"] >= 0


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


def test_rankings(client: TestClient) -> None:
    response = client.get("/rankings", params={"hour": 18, "top_n": 5})

    assert response.status_code == 200
    body = response.json()
    assert body["hour"] == 18
    assert len(body["rankings"]) == 5
    predictions = [r["predicted_avg_daily_ride_count"] for r in body["rankings"]]
    assert predictions == sorted(predictions, reverse=True)
