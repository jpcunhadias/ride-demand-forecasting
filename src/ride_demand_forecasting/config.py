"""Shared constants for the training pipeline and the serving API."""

import os
from pathlib import Path
from typing import Any

# NYC administrative boundary, as fetched once via `osmnx.geocode_to_gdf`
# in the notebook (see notebooks/ride_demand_forecasting.ipynb). Hardcoded
# here so the production pipeline and the API don't depend on a live OSM
# network call.
NYC_LAT_MIN = 40.4766
NYC_LAT_MAX = 40.9176
NYC_LNG_MIN = -74.2588
NYC_LNG_MAX = -73.7002

# Validated via elbow/silhouette analysis on pickup coordinates (see the notebook's
# "Validating the Number of Pickup Zones" section) - highest silhouette score of the
# k=10..80 range tested, replacing the original k=40 visual heuristic.
N_PICKUP_ZONES = 20
RANDOM_STATE = 42
TRAIN_FRACTION = 0.8

FEATURE_ORDER = ["pickup_zone", "hour", "avg_ride_duration_min", "avg_ride_distance"]

# LightGBM hyperparameters tuned via RandomizedSearchCV + TimeSeriesSplit on the
# day-level dataset (see the notebook's "Model Validation" section) - LightGBM edged
# out an identically-tuned XGBoost (MAE 4.626 vs. 4.627) on the same feature set.
LGBM_PARAMS: dict[str, Any] = {
    "n_estimators": 187,
    "num_leaves": 114,
    "learning_rate": 0.055238410897498764,
    "subsample": 0.6571467271687763,
    "colsample_bytree": 0.6624074561769746,
}

RAW_DATA_PATH = Path(os.environ.get("RDF_RAW_DATA_PATH", "data/raw/train.csv"))
MODEL_PATH = Path(os.environ.get("RDF_MODEL_PATH", "models/model.joblib"))
