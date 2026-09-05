from pathlib import Path

from ride_demand_forecasting.data import filter_geo_outliers, load_raw_trips


def test_load_raw_trips_renames_and_derives_duration(raw_trips_csv: Path) -> None:
    df = load_raw_trips(raw_trips_csv)

    assert {"start_time", "start_lat", "start_lng", "end_lat", "end_lng"} <= set(df.columns)
    assert (df["ride_duration_min"] == df["trip_duration"] / 60).all()


def test_filter_geo_outliers_drops_out_of_bounds_rows(raw_trips_csv: Path) -> None:
    df = load_raw_trips(raw_trips_csv)
    n_before = len(df)

    cleaned = filter_geo_outliers(df)

    assert len(cleaned) == n_before - 5
