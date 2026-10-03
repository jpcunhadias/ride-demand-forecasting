"""Loading, cleaning and validating raw trip data."""

from pathlib import Path

import pandas as pd

from ride_demand_forecasting.config import (
    TLC_MAX_DISTANCE_MILES,
    TLC_MAX_DURATION_MIN,
    TLC_MAX_SPEED_MPH,
    TLC_MIN_DAILY_SHARE_OF_MEDIAN,
    TLC_MIN_DISTANCE_MILES,
    TLC_MIN_DURATION_MIN,
)

TLC_RENAME_MAP = {
    "tpep_pickup_datetime": "start_time",
    "tpep_dropoff_datetime": "end_time",
    "PULocationID": "pickup_location_id",
    "DOLocationID": "dropoff_location_id",
    "trip_distance": "trip_distance_miles",
    "fare_amount": "fare_amount",
}
KM_PER_MILE = 1.609344


def load_tlc_trips(
    path: str | Path, month: pd.Period, zone_centroids: pd.DataFrame
) -> pd.DataFrame:
    """Load one month of TLC yellow taxi trips into the pipeline's normalized schema.

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


def validate_daily_coverage(location_daily: pd.DataFrame, month: pd.Period) -> None:
    """Fail if `month` has a day with no trips, or with implausibly few.

    Zone-hour-days without trips are later filled in as zero demand, which is only true
    if the day was actually reported. A gap in the source would otherwise be learned
    as a day when nobody took a taxi.
    """
    daily_totals = location_daily.groupby("date")["ride_count"].sum()
    expected_days = pd.date_range(month.start_time, month.end_time.normalize(), freq="D")

    missing = expected_days.difference(daily_totals.index)
    if len(missing) > 0:
        days = ", ".join(str(day.date()) for day in missing)
        raise ValueError(f"{month} has no trips on: {days}")

    floor = TLC_MIN_DAILY_SHARE_OF_MEDIAN * daily_totals.median()
    too_low = daily_totals[daily_totals < floor]
    if len(too_low) > 0:
        days = ", ".join(f"{day.date()} ({count})" for day, count in too_low.items())
        raise ValueError(f"{month} has implausibly few trips on: {days}")
