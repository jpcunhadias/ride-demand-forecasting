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


def load_raw_trips(path: str | Path) -> pd.DataFrame:
    """Load the raw Kaggle-schema trips CSV into the notebook's normalized schema."""
    df = pd.read_csv(path, parse_dates=["pickup_datetime"])
    df = df.rename(columns=RENAME_MAP)
    df["ride_duration_min"] = df["trip_duration"] / 60
    return df


def filter_geo_outliers(df: pd.DataFrame) -> pd.DataFrame:
    """Drop rides whose pickup or dropoff falls outside the NYC bounding box."""
    in_bounds = (
        df["start_lat"].between(NYC_LAT_MIN, NYC_LAT_MAX)
        & df["start_lng"].between(NYC_LNG_MIN, NYC_LNG_MAX)
        & df["end_lat"].between(NYC_LAT_MIN, NYC_LAT_MAX)
        & df["end_lng"].between(NYC_LNG_MIN, NYC_LNG_MAX)
    )
    return df[in_bounds].copy()
