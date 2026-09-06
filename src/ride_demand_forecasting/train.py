"""End-to-end training pipeline: raw trips -> persisted model artifact.

Ported from notebooks/ride_demand_forecasting.ipynb's baseline-modeling and
Model Validation sections, with fixes and improvements that matter for a
real serving pipeline (the notebook itself is left as the Phase-1
exploratory record, unchanged):

1. The pickup-zone KMeans is fit on train-only coordinates, not train+test
   combined, to avoid leaking test pickup locations into the cluster
   centers.
2. `avg_ride_duration_min` / `avg_ride_distance` aren't accepted as inputs
   to the serving API (a caller can't know them ahead of a future
   prediction window) - a historical (pickup_zone, hour) profile is built
   from training data instead, and stored in the artifact for lookup.
3. Training rows are day-level (one `(pickup_zone, hour, date)` observation
   each) rather than one pre-averaged number per zone-hour for the whole
   split - far more rows (tens of thousands vs. a few hundred) for the
   model to actually learn the conditional mean from, at the cost of the
   target now being noisier per row (which the day-level test metric
   reflects honestly, rather than the old split-averaged number).
4. LightGBM (tuned) replaces the untuned XGBoost baseline - it edged out an
   identically-tuned XGBoost in the notebook's Model Validation comparison.
"""

import logging
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import OrdinalEncoder

from ride_demand_forecasting.clustering import assign_pickup_zones, fit_pickup_zones
from ride_demand_forecasting.config import (
    FEATURE_ORDER,
    LGBM_PARAMS,
    MODEL_PATH,
    N_PICKUP_ZONES,
    RANDOM_STATE,
    RAW_DATA_PATH,
    TRAIN_FRACTION,
)
from ride_demand_forecasting.data import filter_geo_outliers, load_raw_trips
from ride_demand_forecasting.features import (
    add_ride_distance,
    add_temporal_features,
    aggregate_zone_hour,
    aggregate_zone_hour_daily,
    build_zone_hour_profile,
)

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def train(raw_data_path: str | Path = RAW_DATA_PATH) -> dict:
    """Run the full pipeline and return the artifact dict (unsaved)."""
    df = load_raw_trips(raw_data_path)
    df = filter_geo_outliers(df)
    df = add_temporal_features(df)
    df = add_ride_distance(df)

    df = df.sort_values("start_time")
    split_point = int(len(df) * TRAIN_FRACTION)
    train_df, test_df = df.iloc[:split_point].copy(), df.iloc[split_point:].copy()

    kmeans = fit_pickup_zones(train_df)
    train_df["pickup_zone"] = assign_pickup_zones(train_df, kmeans)
    test_df["pickup_zone"] = assign_pickup_zones(test_df, kmeans)

    zones = np.arange(N_PICKUP_ZONES)

    # The static zone-hour profile (historical avg duration/distance, train-only) - both
    # the lookup the service uses at serving time, and the feature source for training.
    agg_train = aggregate_zone_hour(
        train_df, n_days=train_df["start_time"].dt.date.nunique(), zones=zones
    )
    zone_hour_profile = build_zone_hour_profile(agg_train)

    daily_train = aggregate_zone_hour_daily(train_df, zones=zones)
    daily_test = aggregate_zone_hour_daily(test_df, zones=zones)
    daily_train = daily_train.merge(zone_hour_profile, on=["pickup_zone", "hour"], how="left")
    daily_test = daily_test.merge(zone_hour_profile, on=["pickup_zone", "hour"], how="left")

    encoder = OrdinalEncoder()
    X_train = daily_train[FEATURE_ORDER].copy()
    X_test = daily_test[FEATURE_ORDER].copy()
    X_train[["pickup_zone"]] = encoder.fit_transform(X_train[["pickup_zone"]])
    X_test[["pickup_zone"]] = encoder.transform(X_test[["pickup_zone"]])

    y_train = daily_train["ride_count"]
    y_test = daily_test["ride_count"]

    model = LGBMRegressor(**LGBM_PARAMS, random_state=RANDOM_STATE, verbose=-1)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    mae = float(mean_absolute_error(y_test, y_pred))
    rmse = float(np.sqrt(mean_squared_error(y_test, y_pred)))

    return {
        "kmeans": kmeans,
        "encoder": encoder,
        "model": model,
        "zone_hour_profile": zone_hour_profile,
        "zones": zones.tolist(),
        "feature_order": FEATURE_ORDER,
        "metrics": {"mae": mae, "rmse": rmse},
        "trained_at": pd.Timestamp.now("UTC").isoformat(),
        "n_train_rows": len(daily_train),
        "n_test_rows": len(daily_test),
    }


def save_artifact(artifact: dict, model_path: str | Path = MODEL_PATH) -> None:
    model_path = Path(model_path)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(artifact, model_path)


def main() -> None:
    start = time.perf_counter()
    logger.info("Training on %s ...", RAW_DATA_PATH)
    artifact = train()
    save_artifact(artifact)
    elapsed = time.perf_counter() - start
    logger.info("Saved model artifact to %s in %.1fs", MODEL_PATH, elapsed)
    logger.info("Test MAE: %.3f rides/day", artifact["metrics"]["mae"])
    logger.info("Test RMSE: %.3f rides/day", artifact["metrics"]["rmse"])


if __name__ == "__main__":
    main()
