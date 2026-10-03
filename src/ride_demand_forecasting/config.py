"""Shared constants for the training pipeline and the serving API."""

import os
from pathlib import Path
from typing import Any

# NYC administrative boundary, as fetched once via `osmnx.geocode_to_gdf`
# in the notebook (see notebooks/ride_demand_forecasting.ipynb). Hardcoded
# here so the API can reject out-of-area coordinates without a live OSM
# network call.
NYC_LAT_MIN = 40.4766
NYC_LAT_MAX = 40.9176
NYC_LNG_MIN = -74.2588
NYC_LNG_MAX = -73.7002

# Validated via elbow/silhouette analysis on pickup coordinates (see the notebook's
# "Validating the Number of Pickup Zones" section) - highest silhouette score of the
# k=10..80 range tested, replacing the original k=40 visual heuristic. That was on the
# 2016 data's raw coordinates; it still needs re-validating on TLC zone centre points.
N_PICKUP_ZONES = 20
RANDOM_STATE = 42

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
# A day whose total falls below this share of the month's median day is treated as a
# reporting gap, not as a quiet day. Set well below real demand shocks - the 2016
# blizzard in the notebook still reached about a fifth of a normal day.
TLC_MIN_DAILY_SHARE_OF_MEDIAN = 0.05

# Each training run uses the most recent months up to this many.
TLC_TRAIN_WINDOW_MONTHS = 12
# Trip data arrives about two months late, so a model is always serving predictions at
# least that far past the end of its training data. Evaluation leaves the same distance
# between the last training month and the test month.
TLC_EVAL_GAP_MONTHS = 2

# What `ride-demand-backtest` compares by default: these training-window lengths, each
# scored on the same most recent test months.
TLC_BACKTEST_WINDOWS = (3, 6, 12, 24)
TLC_BACKTEST_TEST_MONTHS = 6
# The pickup-zone counts `ride-demand-validate-k` tries by default.
PICKUP_ZONE_CANDIDATES = tuple(range(5, 65, 5))

MODEL_PATH = Path(os.environ.get("RDF_MODEL_PATH", "models/model.joblib"))
# Evaluation metrics of the latest training run, as a small JSON file DVC can compare
# across runs (`dvc metrics diff`).
METRICS_PATH = Path(os.environ.get("RDF_METRICS_PATH", "models/metrics.json"))
MLFLOW_EXPERIMENT = os.environ.get("RDF_MLFLOW_EXPERIMENT", "ride-demand-forecasting")

# If the loaded artifact is older than this, the service logs a warning at startup -
# a cheap, passive signal that a retrain may be overdue. Doesn't block startup or
# serving; there's no scheduled retraining pipeline yet to act on it automatically.
MODEL_STALENESS_WARNING_DAYS = int(os.environ.get("RDF_MODEL_STALENESS_WARNING_DAYS", "30"))
