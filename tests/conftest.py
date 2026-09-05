from pathlib import Path

import numpy as np
import pandas as pd
import pytest

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
