"""End-to-end training pipeline: raw trips -> persisted model artifact.

Ported from notebooks/ride_demand_forecasting.ipynb's baseline-modeling
section, with two fixes that matter for a real serving pipeline (the
notebook itself is left as the Phase-1 exploratory record, unchanged):

1. The pickup-zone KMeans is fit on train-only coordinates, not train+test
   combined, to avoid leaking test pickup locations into the cluster
   centers.
2. `avg_ride_duration_min` / `avg_ride_distance` aren't accepted as inputs
   to the serving API (a caller can't know them ahead of a future
   prediction window) - a historical (pickup_zone, hour) profile is built
   from training data instead, and stored in the artifact for lookup.
"""

import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import OrdinalEncoder
from xgboost import XGBRegressor

from ride_demand_forecasting.clustering import assign_pickup_zones, fit_pickup_zones
from ride_demand_forecasting.config import (
    FEATURE_ORDER,
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
    build_zone_hour_profile,
)


def train(raw_data_path: str | Path = RAW_DATA_PATH) -> dict:
    """Run the full pipeline and return the artifact dict (unsaved)."""
    df = load_raw_trips(raw_data_path)
    df = filter_geo_outliers(df)
    df = add_temporal_features(df)
    df = add_ride_distance(df)

    df = df.sort_values("start_time")
    split_point = int(len(df) * TRAIN_FRACTION)
    train_df, test_df = df.iloc[:split_point], df.iloc[split_point:]

    train_days = train_df["start_time"].dt.date.nunique()
    test_days = test_df["start_time"].dt.date.nunique()

    kmeans = fit_pickup_zones(train_df)
    train_df = train_df.copy()
    test_df = test_df.copy()
    train_df["pickup_zone"] = assign_pickup_zones(train_df, kmeans)
    test_df["pickup_zone"] = assign_pickup_zones(test_df, kmeans)

    zones = np.arange(N_PICKUP_ZONES)
    agg_train = aggregate_zone_hour(train_df, n_days=train_days, zones=zones)
    agg_test = aggregate_zone_hour(test_df, n_days=test_days, zones=zones)

    zone_hour_profile = build_zone_hour_profile(agg_train)

    encoder = OrdinalEncoder()
    X_train = agg_train[FEATURE_ORDER].copy()
    X_test = agg_test[FEATURE_ORDER].copy()
    X_train[["pickup_zone"]] = encoder.fit_transform(X_train[["pickup_zone"]])
    X_test[["pickup_zone"]] = encoder.transform(X_test[["pickup_zone"]])

    y_train = agg_train["avg_daily_ride_count"]
    y_test = agg_test["avg_daily_ride_count"]

    model = XGBRegressor(n_estimators=100, random_state=RANDOM_STATE)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    mae = float(mean_absolute_error(y_test, y_pred))
    rmse = float(np.sqrt(mean_squared_error(y_test, y_pred)))

    return {
        "kmeans": kmeans,
        "encoder": encoder,
        "xgb_model": model,
        "zone_hour_profile": zone_hour_profile,
        "zones": zones.tolist(),
        "feature_order": FEATURE_ORDER,
        "metrics": {"mae": mae, "rmse": rmse},
        "trained_at": pd.Timestamp.now("UTC").isoformat(),
        "n_train_rows": len(train_df),
        "n_test_rows": len(test_df),
    }


def save_artifact(artifact: dict, model_path: str | Path = MODEL_PATH) -> None:
    model_path = Path(model_path)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(artifact, model_path)


def main() -> None:
    start = time.perf_counter()
    print(f"Training on {RAW_DATA_PATH} ...")
    artifact = train()
    save_artifact(artifact)
    elapsed = time.perf_counter() - start
    print(f"Saved model artifact to {MODEL_PATH} in {elapsed:.1f}s")
    print(f"Test MAE: {artifact['metrics']['mae']:.3f} rides/day")
    print(f"Test RMSE: {artifact['metrics']['rmse']:.3f} rides/day")


if __name__ == "__main__":
    main()
