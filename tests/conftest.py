import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import shapefile

from ride_demand_forecasting.config import NYC_LAT_MAX, NYC_LAT_MIN, NYC_LNG_MAX, NYC_LNG_MIN
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


@pytest.fixture
def model_artifact_path(tmp_path: Path, raw_trips_csv: Path) -> Path:
    artifact = train(raw_data_path=raw_trips_csv)
    path = tmp_path / "model.joblib"
    save_artifact(artifact, model_path=path)
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
