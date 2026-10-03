from pathlib import Path

import pandas as pd
import pytest

from ride_demand_forecasting.data import (
    KM_PER_MILE,
    filter_geo_outliers,
    load_raw_trips,
    load_tlc_trips,
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


def test_load_tlc_trips_joins_centre_points_and_drops_unusable_rows(
    tmp_path: Path, tlc_trips: pd.DataFrame, taxi_zones_zip: Path
) -> None:
    path = tmp_path / "yellow_tripdata_2026-08.parquet"
    tlc_trips.to_parquet(path)
    centroids = compute_zone_centroids(taxi_zones_zip)

    df = load_tlc_trips(path, pd.Period("2026-08"), centroids)

    # The stray-month, negative-duration and unmapped-pickup rows are gone; the trip
    # with an unmapped dropoff is kept.
    assert df["pickup_location_id"].tolist() == [1, 2, 3, 2]
    assert df["ride_duration_min"].tolist() == [10, 20, 30, 15]
    assert df["ride_distance_km"].tolist() == pytest.approx(
        [m * KM_PER_MILE for m in (1.0, 2.0, 3.0, 5.0)]
    )
    assert df.loc[0, ["start_lat", "start_lng"]].tolist() == pytest.approx([1.0, 1.0])
    assert df.loc[2, ["start_lat", "start_lng"]].tolist() == pytest.approx([21.0, 23.0])
    assert df["end_lat"].isna().tolist() == [False, False, False, True]
