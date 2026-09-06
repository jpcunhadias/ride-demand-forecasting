"""Feature engineering and zone-hour aggregation."""

from itertools import product

import numpy as np
import pandas as pd

EARTH_RADIUS_KM = 6371.0088


def add_temporal_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["start_time"] = pd.to_datetime(df["start_time"])
    df["hour"] = df["start_time"].dt.hour
    return df


def haversine_distance_km(
    lat1: pd.Series | np.ndarray,
    lng1: pd.Series | np.ndarray,
    lat2: pd.Series | np.ndarray,
    lng2: pd.Series | np.ndarray,
) -> np.ndarray:
    """Vectorized great-circle distance between two sets of coordinates, in km."""
    lat1_r, lng1_r, lat2_r, lng2_r = map(np.radians, (lat1, lng1, lat2, lng2))
    dlat = lat2_r - lat1_r
    dlng = lng2_r - lng1_r
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1_r) * np.cos(lat2_r) * np.sin(dlng / 2) ** 2
    return 2 * EARTH_RADIUS_KM * np.arcsin(np.sqrt(a))


def add_ride_distance(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["ride_distance_km"] = haversine_distance_km(
        df["start_lat"], df["start_lng"], df["end_lat"], df["end_lng"]
    )
    return df


def aggregate_zone_hour(
    df: pd.DataFrame,
    n_days: int,
    zones: np.ndarray | list[int],
    zone_col: str = "pickup_zone",
    time_col: str = "hour",
) -> pd.DataFrame:
    """Aggregate ride-level data by pickup zone and hour, filling missing combos with zero.

    `n_days` is the number of distinct calendar days covered by `df`, used to turn the
    raw per-split ride count into a daily rate (`avg_daily_ride_count`) that's comparable
    across splits covering different numbers of days.
    """
    agg_df = (
        df.groupby([zone_col, time_col])
        .agg(
            ride_count=("ride_duration_min", "count"),
            avg_ride_duration_min=("ride_duration_min", "mean"),
            avg_ride_distance=("ride_distance_km", "mean"),
        )
        .reset_index()
    )

    hours = np.arange(24)
    all_combos = pd.DataFrame(list(product(zones, hours)), columns=[zone_col, time_col])
    agg_df = all_combos.merge(agg_df, on=[zone_col, time_col], how="left")
    agg_df["ride_count"] = agg_df["ride_count"].fillna(0).astype(int)
    agg_df["avg_ride_duration_min"] = agg_df["avg_ride_duration_min"].fillna(0)
    agg_df["avg_ride_distance"] = agg_df["avg_ride_distance"].fillna(0)

    agg_df["avg_daily_ride_count"] = agg_df["ride_count"] / n_days

    return agg_df


def aggregate_zone_hour_daily(
    df: pd.DataFrame,
    zones: np.ndarray | list[int],
    zone_col: str = "pickup_zone",
    time_col: str = "hour",
) -> pd.DataFrame:
    """One row per (zone, hour, date), zero-filled for every combination - so training
    sees genuine zero-demand zone-hour-days, not just observed ones.

    Unlike `aggregate_zone_hour` (one pre-averaged row per zone-hour for a whole split),
    this keeps every day as its own training example - far more rows, and each one a
    natural, un-averaged observation of the target instead of a split-length-dependent
    summary statistic.
    """
    df = df.copy()
    df["date"] = pd.to_datetime(df["start_time"]).dt.date

    daily = df.groupby([zone_col, time_col, "date"]).size().reset_index(name="ride_count")

    dates = df["date"].unique()
    hours = np.arange(24)
    all_combos = pd.DataFrame(
        list(product(zones, hours, dates)), columns=[zone_col, time_col, "date"]
    )
    daily = all_combos.merge(daily, on=[zone_col, time_col, "date"], how="left")
    daily["ride_count"] = daily["ride_count"].fillna(0).astype(int)
    return daily


def build_zone_hour_profile(agg_train: pd.DataFrame) -> pd.DataFrame:
    """The (pickup_zone, hour) -> historical avg duration/distance lookup used at serving
    time, so the API doesn't need the caller to supply features that don't exist yet for a
    future prediction window."""
    return agg_train[["pickup_zone", "hour", "avg_ride_duration_min", "avg_ride_distance"]].copy()
