from pathlib import Path

import pandas as pd
import pytest

from ride_demand_forecasting.data import (
    KM_PER_MILE,
    filter_geo_outliers,
    load_raw_trips,
    load_tlc_trips,
    validate_daily_coverage,
)
from ride_demand_forecasting.zones import compute_zone_centroids


def test_load_raw_trips_renames_and_derives_duration(raw_trips_csv: Path) -> None:
    df = load_raw_trips(raw_trips_csv)

    assert {"start_time", "start_lat", "start_lng", "end_lat", "end_lng"} <= set(df.columns)
    assert (df["ride_duration_min"] == df["trip_duration"] / 60).all()


def test_filter_geo_outliers_drops_out_of_bounds_rows(raw_trips_csv: Path) -> None:
    df = load_raw_trips(raw_trips_csv)
    n_before = len(df)

    cleaned = filter_geo_outliers(df)

    assert len(cleaned) == n_before - 5


@pytest.fixture
def loaded_tlc_trips(tmp_path: Path, tlc_trips: pd.DataFrame, taxi_zones_zip: Path) -> pd.DataFrame:
    path = tmp_path / "yellow_tripdata_2026-08.parquet"
    tlc_trips.to_parquet(path)
    return load_tlc_trips(path, pd.Period("2026-08"), compute_zone_centroids(taxi_zones_zip))


def test_load_tlc_trips_drops_only_rows_that_are_not_countable_pickups(
    loaded_tlc_trips: pd.DataFrame,
) -> None:
    df = loaded_tlc_trips

    assert df["start_time"].dt.day.tolist() == [1, 2, 3, 6, 7, 8, 9, 10, 11, 12, 13]
    assert (df["fare_amount"] >= 0).all()
    assert df["pickup_location_id"].isin([1, 2, 3]).all()


def test_load_tlc_trips_blanks_implausible_measurements_but_keeps_the_trip(
    loaded_tlc_trips: pd.DataFrame,
) -> None:
    df = loaded_tlc_trips
    nan = float("nan")

    # In day order: three clean trips, unmapped dropoff, no dropoff time (distance
    # kept: speed is undefined), no distance, day-long clock error, then three
    # impossible speeds - which blank both values whatever each looks like on its own -
    # and a trip that is only too long.
    assert df["ride_duration_min"].tolist() == pytest.approx(
        [10, 20, 30, 15, nan, 15, nan, nan, nan, nan, nan], nan_ok=True
    )
    assert (df["ride_distance_km"] / KM_PER_MILE).tolist() == pytest.approx(
        [1.0, 2.0, 3.0, 5.0, 2.5, nan, 3.0, nan, nan, nan, 3.0], nan_ok=True
    )


def test_load_tlc_trips_handles_nullable_number_columns(
    tmp_path: Path, tlc_trips: pd.DataFrame, taxi_zones_zip: Path
) -> None:
    # The same trips with distance and fare in pandas' nullable type, plus one trip
    # with neither recorded. Results must not depend on how the file stores numbers.
    unrecorded = tlc_trips.iloc[[0]].assign(
        tpep_pickup_datetime=pd.Timestamp("2026-08-14 20:00"),
        tpep_dropoff_datetime=pd.Timestamp("2026-08-14 20:10"),
        trip_distance=float("nan"),
        fare_amount=float("nan"),
    )
    trips = pd.concat([tlc_trips, unrecorded]).astype(
        {"trip_distance": "Float64", "fare_amount": "Float64"}
    )
    path = tmp_path / "yellow_tripdata_2026-08.parquet"
    trips.to_parquet(path)
    assert str(pd.read_parquet(path)["trip_distance"].dtype) == "Float64"

    df = load_tlc_trips(path, pd.Period("2026-08"), compute_zone_centroids(taxi_zones_zip))
    by_day = df.set_index(df["start_time"].dt.day)

    # No dropoff time: speed is undefined, so the distance stays.
    assert by_day.loc[7, "ride_distance_km"] == pytest.approx(2.5 * KM_PER_MILE)
    assert pd.isna(by_day.loc[7, "ride_duration_min"])
    # Nothing recorded: still a ride, with a usable duration.
    assert by_day.loc[14, "ride_duration_min"] == pytest.approx(10)
    assert pd.isna(by_day.loc[14, "ride_distance_km"])
    assert df["start_time"].dt.day.tolist() == [1, 2, 3, 6, 7, 8, 9, 10, 11, 12, 13, 14]


def test_load_tlc_trips_joins_zone_centre_points(loaded_tlc_trips: pd.DataFrame) -> None:
    df = loaded_tlc_trips

    assert df.loc[0, ["start_lat", "start_lng"]].tolist() == pytest.approx([1.0, 1.0])
    assert df.loc[2, ["start_lat", "start_lng"]].tolist() == pytest.approx([21.0, 23.0])
    # Only the trip dropped off in a zone with no centre point lacks end coordinates.
    assert df.loc[df["end_lat"].isna(), "dropoff_location_id"].tolist() == [265]


def _daily_totals(month: str, totals: dict[int, int] | None = None) -> pd.DataFrame:
    """A location-daily table with 100 rides on every day of `month`, except the days
    overridden in `totals` (0 removes the day altogether)."""
    period = pd.Period(month, freq="M")
    days = pd.date_range(period.start_time, period.end_time.normalize(), freq="D")
    counts = {day: (totals or {}).get(day.day, 100) for day in days}
    return pd.DataFrame(
        {
            "pickup_location_id": 1,
            "date": [day for day, count in counts.items() if count > 0],
            "hour": 8,
            "ride_count": [count for count in counts.values() if count > 0],
        }
    )


def test_validate_daily_coverage_accepts_a_full_month_with_a_quiet_day() -> None:
    # A fifth of a normal day is a real demand shock, not a reporting gap.
    validate_daily_coverage(_daily_totals("2026-08", {23: 20}), pd.Period("2026-08"))


def test_validate_daily_coverage_rejects_a_missing_day() -> None:
    with pytest.raises(ValueError, match="no trips on: 2026-08-15"):
        validate_daily_coverage(_daily_totals("2026-08", {15: 0}), pd.Period("2026-08"))


def test_validate_daily_coverage_rejects_an_implausibly_low_day() -> None:
    with pytest.raises(ValueError, match=r"implausibly few trips on: 2026-08-15 \(2\)"):
        validate_daily_coverage(_daily_totals("2026-08", {15: 2}), pd.Period("2026-08"))
