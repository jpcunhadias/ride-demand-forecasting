import numpy as np
import pandas as pd
import pytest

from ride_demand_forecasting.clustering import assign_pickup_zones, fit_pickup_zones
from ride_demand_forecasting.config import N_PICKUP_ZONES


def _coords_df(n: int) -> pd.DataFrame:
    rng = np.random.default_rng(1)
    return pd.DataFrame(
        {
            "start_lat": rng.uniform(40.5, 40.9, size=n),
            "start_lng": rng.uniform(-74.2, -73.7, size=n),
        }
    )


def test_fit_pickup_zones_produces_expected_number_of_clusters() -> None:
    kmeans = fit_pickup_zones(_coords_df(500))

    assert kmeans.cluster_centers_.shape == (N_PICKUP_ZONES, 2)


def test_assign_pickup_zones_returns_valid_zone_ids() -> None:
    train = _coords_df(500)
    test = _coords_df(50)
    kmeans = fit_pickup_zones(train)

    zones = assign_pickup_zones(test, kmeans)

    assert len(zones) == len(test)
    assert set(zones) <= set(range(N_PICKUP_ZONES))


def test_fit_pickup_zones_weighted_points_match_repeated_rows() -> None:
    points = pd.DataFrame({"start_lat": [0.0, 1.0, 10.0], "start_lng": [0.0, 0.0, 0.0]})
    weights = np.array([1, 3, 2])

    weighted = fit_pickup_zones(points, n_zones=2, sample_weight=weights)
    repeated = fit_pickup_zones(points.loc[points.index.repeat(weights)], n_zones=2)

    def sorted_centers(kmeans):
        return np.array(sorted(kmeans.cluster_centers_.tolist()))

    # The first two points share a cluster, pulled towards the heavier one.
    assert sorted_centers(weighted) == pytest.approx(sorted_centers(repeated))
    assert sorted_centers(weighted)[0] == pytest.approx([0.75, 0.0])
