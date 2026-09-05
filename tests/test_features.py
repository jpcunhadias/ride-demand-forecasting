import numpy as np
import pandas as pd
import pytest

from ride_demand_forecasting.features import (
    add_temporal_features,
    aggregate_zone_hour,
    build_zone_hour_profile,
    haversine_distance_km,
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
