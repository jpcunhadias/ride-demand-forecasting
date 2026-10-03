import numpy as np
import pandas as pd
import pytest

from ride_demand_forecasting.features import (
    add_temporal_features,
    aggregate_location_hour_daily,
    aggregate_zone_hour,
    aggregate_zone_hour_daily,
    build_zone_hour_profile,
    haversine_distance_km,
    zone_hour_daily_counts,
    zone_hour_profile,
)


def test_add_temporal_features_extracts_hour() -> None:
    df = pd.DataFrame({"start_time": ["2016-01-01 08:30:00", "2016-01-01 23:00:00"]})

    result = add_temporal_features(df)

    assert result["hour"].tolist() == [8, 23]


def test_haversine_distance_km_zero_for_identical_points() -> None:
    d = haversine_distance_km(
        np.array([40.7]), np.array([-74.0]), np.array([40.7]), np.array([-74.0])
    )
    assert d[0] == pytest.approx(0.0, abs=1e-9)


def test_haversine_distance_km_one_degree_at_equator() -> None:
    # At the equator, cos(lat) == 1, so 1 degree of longitude is exactly
    # radius * radians(1) of great-circle distance.
    d = haversine_distance_km(np.array([0.0]), np.array([0.0]), np.array([0.0]), np.array([1.0]))
    expected = 6371.0088 * np.radians(1.0)
    assert d[0] == pytest.approx(expected, rel=1e-6)


def test_aggregate_zone_hour_fills_missing_combos_and_computes_daily_rate() -> None:
    df = pd.DataFrame(
        {
            "pickup_zone": [0, 0, 1],
            "hour": [5, 5, 5],
            "ride_duration_min": [10.0, 20.0, 30.0],
            "ride_distance_km": [1.0, 2.0, 3.0],
        }
    )

    agg = aggregate_zone_hour(df, n_days=2, zones=[0, 1])

    assert len(agg) == 2 * 24  # every (zone, hour) combo present

    zone0_hour5 = agg[(agg["pickup_zone"] == 0) & (agg["hour"] == 5)].iloc[0]
    assert zone0_hour5["ride_count"] == 2
    assert zone0_hour5["avg_ride_duration_min"] == pytest.approx(15.0)
    assert zone0_hour5["avg_daily_ride_count"] == pytest.approx(1.0)  # 2 rides / 2 days

    zone1_hour6 = agg[(agg["pickup_zone"] == 1) & (agg["hour"] == 6)].iloc[0]
    assert zone1_hour6["ride_count"] == 0
    assert zone1_hour6["avg_ride_duration_min"] == 0
    assert zone1_hour6["avg_daily_ride_count"] == 0


def test_aggregate_zone_hour_counts_trips_with_missing_measurements() -> None:
    df = pd.DataFrame(
        {
            "pickup_zone": [0, 0, 0],
            "hour": [8, 8, 8],
            "ride_duration_min": [10.0, np.nan, 20.0],
            "ride_distance_km": [2.0, 4.0, np.nan],
        }
    )

    row = aggregate_zone_hour(df, n_days=1, zones=[0]).query("hour == 8").iloc[0]

    assert row["ride_count"] == 3
    assert row["avg_ride_duration_min"] == pytest.approx(15.0)
    assert row["avg_ride_distance"] == pytest.approx(3.0)


def test_aggregate_zone_hour_daily_zero_fills_every_zone_hour_date_combo() -> None:
    df = pd.DataFrame(
        {
            "pickup_zone": [0, 0, 1],
            "hour": [5, 5, 6],
            "start_time": [
                "2016-01-01 05:10:00",
                "2016-01-01 05:40:00",
                "2016-01-02 06:15:00",
            ],
        }
    )

    daily = aggregate_zone_hour_daily(df, zones=[0, 1])

    # 2 zones x 24 hours x 2 distinct dates
    assert len(daily) == 2 * 24 * 2

    zone0_hour5_jan1 = daily[
        (daily["pickup_zone"] == 0)
        & (daily["hour"] == 5)
        & (daily["date"] == pd.Timestamp("2016-01-01").date())
    ].iloc[0]
    assert zone0_hour5_jan1["ride_count"] == 2

    zone0_hour5_jan2 = daily[
        (daily["pickup_zone"] == 0)
        & (daily["hour"] == 5)
        & (daily["date"] == pd.Timestamp("2016-01-02").date())
    ].iloc[0]
    assert zone0_hour5_jan2["ride_count"] == 0


def test_build_zone_hour_profile_has_expected_columns() -> None:
    agg_train = aggregate_zone_hour(
        pd.DataFrame(
            {
                "pickup_zone": [0],
                "hour": [0],
                "ride_duration_min": [10.0],
                "ride_distance_km": [1.0],
            }
        ),
        n_days=1,
        zones=[0],
    )

    profile = build_zone_hour_profile(agg_train)

    assert list(profile.columns) == [
        "pickup_zone",
        "hour",
        "avg_ride_duration_min",
        "avg_ride_distance",
    ]


def _location_daily(rows: list[tuple]) -> pd.DataFrame:
    """Rows are (location, date, hour, rides, duration_sum, duration_n, distance_sum,
    distance_n)."""
    df = pd.DataFrame(
        rows,
        columns=[
            "pickup_location_id",
            "date",
            "hour",
            "ride_count",
            "duration_sum",
            "duration_n",
            "distance_sum",
            "distance_n",
        ],
    )
    df["date"] = pd.to_datetime(df["date"])
    return df


def test_aggregate_location_hour_daily_tracks_usable_measurements_separately() -> None:
    trips = pd.DataFrame(
        {
            "pickup_location_id": [7, 7, 7, 9],
            "start_time": pd.to_datetime(
                ["2026-08-01 08:05", "2026-08-01 08:40", "2026-08-01 08:55", "2026-08-02 23:10"]
            ),
            "ride_duration_min": [10.0, np.nan, 20.0, 5.0],
            "ride_distance_km": [2.0, 4.0, np.nan, np.nan],
        }
    )

    result = aggregate_location_hour_daily(trips)

    assert len(result) == 2
    busy = result[result["pickup_location_id"] == 7].iloc[0]
    assert (busy["date"], busy["hour"]) == (pd.Timestamp("2026-08-01"), 8)
    assert busy["ride_count"] == 3
    assert (busy["duration_sum"], busy["duration_n"]) == (30.0, 2)
    assert (busy["distance_sum"], busy["distance_n"]) == (6.0, 2)
    quiet = result[result["pickup_location_id"] == 9].iloc[0]
    assert (quiet["ride_count"], quiet["distance_n"]) == (1, 0)


def test_zone_hour_daily_counts_merges_locations_and_zero_fills() -> None:
    location_daily = _location_daily(
        [
            (1, "2026-08-01", 8, 3, 0, 0, 0, 0),
            (2, "2026-08-01", 8, 4, 0, 0, 0, 0),
            (3, "2026-08-02", 9, 5, 0, 0, 0, 0),
        ]
    )
    zone_map = pd.Series({1: 0, 2: 0, 3: 1})

    result = zone_hour_daily_counts(location_daily, zone_map, zones=[0, 1])

    assert len(result) == 2 * 24 * 2
    counts = result.set_index(["pickup_zone", "hour", "date"])["ride_count"]
    assert counts[(0, 8, pd.Timestamp("2026-08-01"))] == 7
    assert counts[(1, 9, pd.Timestamp("2026-08-02"))] == 5
    assert counts[(0, 8, pd.Timestamp("2026-08-02"))] == 0
    assert counts.sum() == 12


def test_zone_hour_profile_falls_back_to_zone_then_overall_average() -> None:
    location_daily = _location_daily(
        [
            # Zone 0: measurements at 8h (two locations) and 9h, a ride at 10h with none.
            (1, "2026-08-01", 8, 2, 20.0, 2, 6.0, 2),
            (2, "2026-08-01", 8, 2, 40.0, 2, 2.0, 2),
            (1, "2026-08-01", 9, 1, 30.0, 1, 7.0, 1),
            (1, "2026-08-01", 10, 1, 0.0, 0, 0.0, 0),
            # Zone 1: a ride, but nothing measured anywhere.
            (3, "2026-08-01", 8, 1, 0.0, 0, 0.0, 0),
        ]
    )
    zone_map = pd.Series({1: 0, 2: 0, 3: 1})

    profile = zone_hour_profile(location_daily, zone_map, zones=[0, 1]).set_index(
        ["pickup_zone", "hour"]
    )

    assert len(profile) == 2 * 24
    assert not profile.isna().any().any()
    assert profile.loc[(0, 8), "avg_ride_duration_min"] == pytest.approx(15.0)
    assert profile.loc[(0, 9), "avg_ride_distance"] == pytest.approx(7.0)
    # No measurement at 10h or 3h: zone 0's average over all its measured trips.
    assert profile.loc[(0, 10), "avg_ride_duration_min"] == pytest.approx(18.0)
    assert profile.loc[(0, 3), "avg_ride_distance"] == pytest.approx(3.0)
    # Zone 1 has none at all: the overall average.
    assert profile.loc[(1, 8), "avg_ride_duration_min"] == pytest.approx(18.0)
