"""Loading and geo-cleaning raw trip data."""

from pathlib import Path

import pandas as pd

from ride_demand_forecasting.config import (
    NYC_LAT_MAX,
    NYC_LAT_MIN,
    NYC_LNG_MAX,
    NYC_LNG_MIN,
    TLC_MAX_DISTANCE_MILES,
    TLC_MAX_DURATION_MIN,
    TLC_MAX_SPEED_MPH,
    TLC_MIN_DISTANCE_MILES,
    TLC_MIN_DURATION_MIN,
)

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
    `zones.compute_zone_centroids`).

    Cleaning keeps two questions apart, because most bad rows are real trips with one
    bad measurement:

    - *Was this a pickup we should count?* A row is dropped only if it starts outside
      `month` (the published files carry a few stray timestamps), has a negative fare
      (a voided record, nearly always the twin of a positive one), is a false start
      (under a minute and no distance), or was picked up in a zone with no centre
      point (the TLC's "unknown" and "outside NYC" IDs).
    - *Can its measurements be trusted?* An implausible duration or distance is set to
      missing rather than dropping the trip, so it still counts as demand but stays out
      of the duration/distance averages. An implausible speed blanks both, since there's
      no telling which of the two is wrong.

    An unmapped dropoff only leaves `end_lat`/`end_lng` empty.
    """
    df = pd.read_parquet(path, columns=list(TLC_RENAME_MAP)).rename(columns=TLC_RENAME_MAP)
    # Plain floats, whatever the file stores: with pandas' nullable types a comparison
    # against a missing value is itself missing, which the rules below would then
    # treat as a failed check instead of an inapplicable one.
    df = df.astype({"trip_distance_miles": "float64", "fare_amount": "float64"})
    df = df[df["start_time"].dt.to_period("M") == month]
    df = df[~(df["fare_amount"] < 0)]

    duration = (df["end_time"] - df["start_time"]).dt.total_seconds() / 60
    miles = df.pop("trip_distance_miles")
    false_start = (duration < TLC_MIN_DURATION_MIN) & (miles == 0)
    df, duration, miles = df[~false_start], duration[~false_start], miles[~false_start]

    duration_ok = duration.between(TLC_MIN_DURATION_MIN, TLC_MAX_DURATION_MIN)
    distance_ok = miles.between(TLC_MIN_DISTANCE_MILES, TLC_MAX_DISTANCE_MILES)
    # Judged on the raw values, so an impossible speed blanks both measurements even
    # when one of them is already out of range on its own. Speed is undefined without a
    # positive duration, which leaves a trip with no dropoff time its distance.
    too_fast = miles / (duration.where(duration > 0) / 60) > TLC_MAX_SPEED_MPH
    df["ride_duration_min"] = duration.where(duration_ok & ~too_fast)
    df["ride_distance_km"] = (miles * KM_PER_MILE).where(distance_ok & ~too_fast)

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
