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

# NYC TLC yellow taxi trip records (one Parquet file per month, published roughly two
# months in arrears) and the taxi zone boundary file, both served from the TLC's public
# CDN - see https://www.nyc.gov/site/tlc/about/tlc-trip-record-data.page
TLC_BASE_URL = os.environ.get("RDF_TLC_BASE_URL", "https://d37ci6vzurychx.cloudfront.net")
TLC_DATA_DIR = Path(os.environ.get("RDF_TLC_DATA_DIR", "data/raw/tlc"))
# How far back `ride-demand-ingest` looks when no start month is given: a 12-month
# training window plus the publication lag.
TLC_DEFAULT_LOOKBACK_MONTHS = 14

# Plausibility limits for a trip's measurements, in the TLC's own units. A value outside
# them is treated as missing rather than as a reason to drop the trip (see
# `data.load_tlc_trips`). Chosen from the August 2026 file, where 97.6% of trips last
# 1-180 minutes, 95.7% cover 0.1-60 miles, and the 99.9th percentile speed is 50 mph.
TLC_MIN_DURATION_MIN = 1
TLC_MAX_DURATION_MIN = 180
TLC_MIN_DISTANCE_MILES = 0.1
TLC_MAX_DISTANCE_MILES = 60
TLC_MAX_SPEED_MPH = 70

RAW_DATA_PATH = Path(os.environ.get("RDF_RAW_DATA_PATH", "data/raw/train.csv"))
MODEL_PATH = Path(os.environ.get("RDF_MODEL_PATH", "models/model.joblib"))

# If the loaded artifact is older than this, the service logs a warning at startup -
# a cheap, passive signal that a retrain may be overdue. Doesn't block startup or
# serving; there's no scheduled retraining pipeline yet to act on it automatically.
MODEL_STALENESS_WARNING_DAYS = int(os.environ.get("RDF_MODEL_STALENESS_WARNING_DAYS", "30"))
