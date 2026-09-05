"""Loads a trained artifact once and serves predictions from it."""

from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from ride_demand_forecasting.config import MODEL_PATH


class UnknownZoneError(ValueError):
    pass


class PredictionService:
    def __init__(self, model_path: str | Path = MODEL_PATH):
        artifact = joblib.load(model_path)
        self.kmeans = artifact["kmeans"]
        self.encoder = artifact["encoder"]
        self.model = artifact["xgb_model"]
        self.feature_order = artifact["feature_order"]
        self.zones: list[int] = artifact["zones"]
        self.metrics: dict = artifact["metrics"]

        profile = artifact["zone_hour_profile"]
        self._profile: dict[tuple[int, int], tuple[float, float]] = {
            (int(row.pickup_zone), int(row.hour)): (
                float(row.avg_ride_duration_min),
                float(row.avg_ride_distance),
            )
            for row in profile.itertuples()
        }

    def zone_for_coords(self, lat: float, lng: float) -> int:
        coords = pd.DataFrame({"start_lat": [lat], "start_lng": [lng]})
        return int(self.kmeans.predict(coords)[0])

    def _profile_for(self, pickup_zone: int, hour: int) -> tuple[float, float]:
        return self._profile.get((pickup_zone, hour), (0.0, 0.0))

    def _predict_batch(self, pickup_zones: list[int], hour: int) -> np.ndarray:
        avg_duration, avg_distance = zip(
            *(self._profile_for(zone, hour) for zone in pickup_zones), strict=True
        )
        features = pd.DataFrame(
            {
                "pickup_zone": pickup_zones,
                "hour": hour,
                "avg_ride_duration_min": avg_duration,
                "avg_ride_distance": avg_distance,
            }
        )
        features[["pickup_zone"]] = self.encoder.transform(features[["pickup_zone"]])
        return self.model.predict(features[self.feature_order])

    def predict_zone(self, pickup_zone: int, hour: int) -> float:
        if pickup_zone not in self.zones:
            raise UnknownZoneError(f"Unknown pickup_zone: {pickup_zone}")
        return float(self._predict_batch([pickup_zone], hour)[0])

    def predict_coords(self, lat: float, lng: float, hour: int) -> tuple[int, float]:
        zone = self.zone_for_coords(lat, lng)
        return zone, self.predict_zone(zone, hour)

    def rank(self, hour: int, top_n: int | None = None) -> list[tuple[int, float]]:
        predictions = self._predict_batch(self.zones, hour)
        ranked = sorted(zip(self.zones, predictions, strict=True), key=lambda p: -p[1])
        return [(zone, float(pred)) for zone, pred in ranked[:top_n]]
