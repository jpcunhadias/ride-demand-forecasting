"""Loading and geo-cleaning raw trip data."""

from pathlib import Path

import pandas as pd

from ride_demand_forecasting.config import NYC_LAT_MAX, NYC_LAT_MIN, NYC_LNG_MAX, NYC_LNG_MIN

RENAME_MAP = {
    "pickup_datetime": "start_time",
    "pickup_latitude": "start_lat",
    "pickup_longitude": "start_lng",
    "dropoff_latitude": "end_lat",
    "dropoff_longitude": "end_lng",
}

TLC_RENAME_MAP = {
    "tpep_pickup_datetime": "start_time",
    "tpep_dropoff_datetime": "end_time",
    "PULocationID": "pickup_location_id",
    "DOLocationID": "dropoff_location_id",
    "trip_distance": "trip_distance_miles",
    "fare_amount": "fare_amount",
}
KM_PER_MILE = 1.609344


def load_raw_trips(path: str | Path) -> pd.DataFrame:
    """Load the raw Kaggle-schema trips CSV into the notebook's normalized schema."""
    df = pd.read_csv(path, parse_dates=["pickup_datetime"])
    df = df.rename(columns=RENAME_MAP)
    df["ride_duration_min"] = df["trip_duration"] / 60
    return df


def load_tlc_trips(
    path: str | Path, month: pd.Period, zone_centroids: pd.DataFrame
) -> pd.DataFrame:
    """Load one month of TLC yellow taxi trips into the same normalized schema.

    Pickup and dropoff zone IDs are replaced by that zone's centre point (see
    `zones.compute_zone_centroids`). Trips are dropped if they start outside `month`
    (the published files carry a few stray timestamps), have a non-positive duration,
    or were picked up in a zone with no centre point (the TLC's "unknown" and "outside
    NYC" IDs). An unmapped dropoff only leaves `end_lat`/`end_lng` empty - the pickup
    still counts as demand.
    """
    df = pd.read_parquet(path, columns=list(TLC_RENAME_MAP)).rename(columns=TLC_RENAME_MAP)
    df = df[df["start_time"].dt.to_period("M") == month]

    df["ride_duration_min"] = (df["end_time"] - df["start_time"]).dt.total_seconds() / 60
    df = df[df["ride_duration_min"] > 0]
    df["ride_distance_km"] = df.pop("trip_distance_miles") * KM_PER_MILE

    coords = zone_centroids[["location_id", "lat", "lng"]]
    df = df.merge(
        coords.rename(
            columns={"location_id": "pickup_location_id", "lat": "start_lat", "lng": "start_lng"}
        ),
        on="pickup_location_id",
        how="inner",
    )
    df = df.merge(
        coords.rename(
            columns={"location_id": "dropoff_location_id", "lat": "end_lat", "lng": "end_lng"}
        ),
        on="dropoff_location_id",
        how="left",
    )
    return df.drop(columns="end_time").sort_values("start_time").reset_index(drop=True)


def filter_geo_outliers(df: pd.DataFrame) -> pd.DataFrame:
    """Drop rides whose pickup or dropoff falls outside the NYC bounding box."""
    in_bounds = (
        df["start_lat"].between(NYC_LAT_MIN, NYC_LAT_MAX)
        & df["start_lng"].between(NYC_LNG_MIN, NYC_LNG_MAX)
        & df["end_lat"].between(NYC_LAT_MIN, NYC_LAT_MAX)
        & df["end_lng"].between(NYC_LNG_MIN, NYC_LNG_MAX)
    )
    return df[in_bounds].copy()
