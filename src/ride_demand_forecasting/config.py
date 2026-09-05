"""Shared constants for the training pipeline and the serving API."""

import os
from pathlib import Path

# NYC administrative boundary, as fetched once via `osmnx.geocode_to_gdf`
# in the notebook (see notebooks/ride_demand_forecasting.ipynb). Hardcoded
# here so the production pipeline and the API don't depend on a live OSM
# network call.
NYC_LAT_MIN = 40.4766
NYC_LAT_MAX = 40.9176
NYC_LNG_MIN = -74.2588
NYC_LNG_MAX = -73.7002

N_PICKUP_ZONES = 40
RANDOM_STATE = 42
TRAIN_FRACTION = 0.8

FEATURE_ORDER = ["pickup_zone", "hour", "avg_ride_duration_min", "avg_ride_distance"]

RAW_DATA_PATH = Path(os.environ.get("RDF_RAW_DATA_PATH", "data/raw/train.csv"))
MODEL_PATH = Path(os.environ.get("RDF_MODEL_PATH", "models/model.joblib"))
