from pathlib import Path

import pytest

from ride_demand_forecasting.zones import CENTROID_COLUMNS, compute_zone_centroids


def test_compute_zone_centroids_returns_one_row_per_location_id(taxi_zones_zip: Path) -> None:
    centroids = compute_zone_centroids(taxi_zones_zip)

    assert list(centroids.columns) == CENTROID_COLUMNS
    assert centroids["location_id"].tolist() == [1, 2, 3]
    assert centroids["zone"].tolist() == ["Plain", "Holed", "Split"]


def test_compute_zone_centroids_handles_holes_and_split_zones(taxi_zones_zip: Path) -> None:
    centroids = compute_zone_centroids(taxi_zones_zip).set_index("location_id")

    assert centroids.loc[1, ["lng", "lat"]].tolist() == pytest.approx([1.0, 1.0])
    # 4x4 square centred on (12, 12) minus a 1x1 hole centred on (11.5, 11.5).
    holed = (16 * 12 - 1 * 11.5) / 15
    assert centroids.loc[2, ["lng", "lat"]].tolist() == pytest.approx([holed, holed])
    # Two equal squares centred on (21, 21) and (25, 21).
    assert centroids.loc[3, ["lng", "lat"]].tolist() == pytest.approx([23.0, 21.0])
