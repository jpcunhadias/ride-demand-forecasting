import zipfile
from collections.abc import Callable
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import shapefile

from ride_demand_forecasting.config import NYC_LAT_MAX, NYC_LAT_MIN, NYC_LNG_MAX, NYC_LNG_MIN
from ride_demand_forecasting.ingest import CENTROIDS_FILENAME, trip_path
from ride_demand_forecasting.train import save_artifact, train

N_ROWS = 600


@pytest.fixture
def raw_trips_csv(tmp_path: Path) -> Path:
    """A tiny synthetic dataset with the real Kaggle column names, spread over
    several days so a chronological train/test split has data on both sides,
    and a few coordinates deliberately outside the NYC bbox to exercise the
    geo-outlier filter.
    """
    rng = np.random.default_rng(0)

    start_times = pd.date_range("2016-01-01", periods=N_ROWS, freq="27min")
    lat = rng.uniform(NYC_LAT_MIN + 0.05, NYC_LAT_MAX - 0.05, size=N_ROWS)
    lng = rng.uniform(NYC_LNG_MIN + 0.05, NYC_LNG_MAX - 0.05, size=N_ROWS)
    end_lat = rng.uniform(NYC_LAT_MIN + 0.05, NYC_LAT_MAX - 0.05, size=N_ROWS)
    end_lng = rng.uniform(NYC_LNG_MIN + 0.05, NYC_LNG_MAX - 0.05, size=N_ROWS)

    # A handful of GPS-glitch rows, clearly outside the NYC bbox.
    lat[:5] = 34.0
    lng[:5] = -121.9

    df = pd.DataFrame(
        {
            "id": [f"id{i}" for i in range(N_ROWS)],
            "vendor_id": rng.integers(1, 3, size=N_ROWS),
            "pickup_datetime": start_times,
            "dropoff_datetime": start_times + pd.to_timedelta(rng.integers(60, 1800, N_ROWS), "s"),
            "passenger_count": rng.integers(1, 5, size=N_ROWS),
            "pickup_longitude": lng,
            "pickup_latitude": lat,
            "dropoff_longitude": end_lng,
            "dropoff_latitude": end_lat,
            "store_and_fwd_flag": "N",
            "trip_duration": rng.integers(60, 1800, size=N_ROWS),
        }
    )

    path = tmp_path / "train.csv"
    df.to_csv(path, index=False)
    return path


WGS84_PRJ = (
    'GEOGCS["GCS_WGS_1984",DATUM["D_WGS_1984",'
    'SPHEROID["WGS_1984",6378137.0,298.257223563]],'
    'PRIMEM["Greenwich",0.0],UNIT["Degree",0.0174532925199433]]'
)


def _square(x: float, y: float, size: float, clockwise: bool = True) -> list[list[float]]:
    ring = [[x, y], [x, y + size], [x + size, y + size], [x + size, y], [x, y]]
    return ring if clockwise else ring[::-1]


@pytest.fixture
def taxi_zones_zip(tmp_path: Path) -> Path:
    """A tiny zipped shapefile laid out like the TLC's `taxi_zones.zip`, already in
    WGS84 so expected centre points can be read straight off the coordinates:

    - zone 1: a plain 2x2 square
    - zone 2: a 4x4 square with a 1x1 hole
    - zone 3: split across two separate shapes sharing the same LocationID
    """
    shp_dir = tmp_path / "taxi_zones"
    shp_dir.mkdir()
    with shapefile.Writer(str(shp_dir / "taxi_zones")) as writer:
        writer.field("LocationID", "N")
        writer.field("zone", "C")
        writer.field("borough", "C")

        writer.poly([_square(0, 0, 2)])
        writer.record(1, "Plain", "Queens")
        writer.poly([_square(10, 10, 4), _square(11, 11, 1, clockwise=False)])
        writer.record(2, "Holed", "Bronx")
        writer.poly([_square(20, 20, 2)])
        writer.record(3, "Split", "Manhattan")
        writer.poly([_square(24, 20, 2)])
        writer.record(3, "Split", "Manhattan")
    (shp_dir / "taxi_zones.prj").write_text(WGS84_PRJ)

    zip_path = tmp_path / "taxi_zones.zip"
    with zipfile.ZipFile(zip_path, "w") as archive:
        for file in sorted(shp_dir.iterdir()):
            archive.write(file, f"taxi_zones/{file.name}")
    return zip_path


@pytest.fixture
def tlc_trips() -> pd.DataFrame:
    """A few rows in the TLC yellow taxi schema for August 2026: three clean trips,
    plus one each of the cases `load_tlc_trips` has to handle. Each row is
    (pickup, minutes, miles, pickup zone, dropoff zone, fare)."""
    rows = [
        ("2026-08-01 08:00", 10, 1.0, 1, 2, 8.0),
        ("2026-08-02 09:30", 20, 2.0, 2, 3, 15.0),
        ("2026-08-03 18:15", 30, 3.0, 3, 1, 22.0),
        # Dropped: not a pickup to count.
        ("2026-07-31 23:50", 10, 1.0, 1, 2, 8.0),  # stray timestamp from another month
        ("2026-08-01 08:00", 10, 1.0, 1, 2, -8.0),  # voided twin of the first trip
        ("2026-08-04 10:00", 0.5, 0.0, 1, 1, 3.0),  # false start
        ("2026-08-05 11:00", 10, 1.0, 264, 1, 8.0),  # pickup zone with no centre point
        # Kept, with a measurement blanked.
        ("2026-08-06 12:00", 15, 5.0, 2, 265, 30.0),  # dropoff zone with no centre point
        ("2026-08-07 13:00", 0, 2.5, 1, 2, 12.0),  # vendor reports no dropoff time
        ("2026-08-08 14:00", 15, 0.0, 2, 3, 14.0),  # no distance recorded
        ("2026-08-09 15:00", 1439, 3.0, 3, 1, 20.0),  # clock error, a day long
        ("2026-08-10 16:00", 20, 500.0, 1, 2, 9.0),  # impossible distance, and speed
        ("2026-08-11 17:00", 10, 30.0, 2, 3, 70.0),  # each value plausible, 180 mph is not
        ("2026-08-12 18:00", 0.5, 5.0, 3, 1, 25.0),  # too short, and 600 mph
        ("2026-08-13 19:00", 200, 3.0, 1, 2, 18.0),  # too long, distance still usable
    ]
    pickups = pd.to_datetime([row[0] for row in rows])
    return pd.DataFrame(
        {
            "VendorID": 1,
            "tpep_pickup_datetime": pickups,
            "tpep_dropoff_datetime": pickups + pd.to_timedelta([row[1] for row in rows], "min"),
            "trip_distance": [row[2] for row in rows],
            "PULocationID": [row[3] for row in rows],
            "DOLocationID": [row[4] for row in rows],
            "fare_amount": [row[5] for row in rows],
        }
    )


TLC_FIXTURE_MONTHS = ["2026-04", "2026-05", "2026-06", "2026-07", "2026-08"]
N_TLC_FIXTURE_LOCATIONS = 30
TRIPS_PER_DAY = 150


def synthetic_zone_centroids() -> pd.DataFrame:
    """Centre points for a 6x5 grid of made-up taxi zones inside the NYC bbox."""
    lat, lng = np.meshgrid(np.linspace(40.55, 40.85, 6), np.linspace(-74.15, -73.75, 5))
    location_ids = np.arange(1, N_TLC_FIXTURE_LOCATIONS + 1)
    return pd.DataFrame(
        {
            "location_id": location_ids,
            "zone": [f"Zone {i}" for i in location_ids],
            "borough": "Test",
            "lat": lat.ravel(),
            "lng": lng.ravel(),
        }
    )


def synthetic_tlc_month(month: str, rng: np.random.Generator) -> pd.DataFrame:
    """A month of clean trips in the TLC schema, with every day covered and demand
    skewed towards the low-numbered zones."""
    period = pd.Period(month, freq="M")
    days = pd.date_range(period.start_time, period.end_time.normalize(), freq="D")
    n = len(days) * TRIPS_PER_DAY
    pickups = np.repeat(days, TRIPS_PER_DAY) + pd.to_timedelta(rng.integers(0, 86400, n), "s")
    location_ids = np.arange(1, N_TLC_FIXTURE_LOCATIONS + 1)
    weights = 1 / location_ids
    return pd.DataFrame(
        {
            "VendorID": 1,
            "tpep_pickup_datetime": pickups,
            "tpep_dropoff_datetime": pickups + pd.to_timedelta(rng.integers(300, 2400, n), "s"),
            "trip_distance": rng.uniform(0.5, 8.0, n),
            "PULocationID": rng.choice(location_ids, n, p=weights / weights.sum()),
            "DOLocationID": rng.choice(location_ids, n),
            "fare_amount": rng.uniform(5.0, 60.0, n),
        }
    )


def write_tlc_data_dir(data_dir: Path, months: list[str]) -> Path:
    data_dir.mkdir(parents=True, exist_ok=True)
    synthetic_zone_centroids().to_csv(data_dir / CENTROIDS_FILENAME, index=False)
    rng = np.random.default_rng(0)
    for month in months:
        synthetic_tlc_month(month, rng).to_parquet(trip_path(pd.Period(month), data_dir))
    return data_dir


@pytest.fixture
def make_tlc_data_dir() -> Callable[[Path, list[str]], Path]:
    return write_tlc_data_dir


@pytest.fixture(scope="session")
def tlc_data_dir(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """An ingested-data directory as `ride-demand-ingest` would leave it: the zone
    centre-point lookup plus five consecutive monthly trip files."""
    return write_tlc_data_dir(tmp_path_factory.mktemp("tlc"), TLC_FIXTURE_MONTHS)


@pytest.fixture(scope="session")
def trained_artifact(tlc_data_dir: Path) -> dict:
    """Trained once per test session. Treat it as read-only - copy before changing it."""
    return train(data_dir=tlc_data_dir)


@pytest.fixture(scope="session")
def model_artifact_path(tmp_path_factory: pytest.TempPathFactory, trained_artifact: dict) -> Path:
    path = tmp_path_factory.mktemp("model") / "model.joblib"
    save_artifact(trained_artifact, model_path=path)
    return path
