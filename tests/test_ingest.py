import shutil
import urllib.error
from email.message import Message
from pathlib import Path

import pandas as pd
import pytest

from ride_demand_forecasting.ingest import (
    CENTROIDS_FILENAME,
    Download,
    ingest_trips,
    ingest_zone_centroids,
    trip_path,
)


def make_download(
    trips: pd.DataFrame, published: set[str], status_when_missing: int = 403
) -> tuple[Download, list[str]]:
    """A stand-in for the CDN: serves `trips` for published months, an HTTP error
    for the rest, and records every URL it was asked for."""
    requested: list[str] = []

    def download(url: str, dest: Path) -> None:
        requested.append(url)
        month = url.rsplit("_", 1)[-1].removesuffix(".parquet")
        if month not in published:
            raise urllib.error.HTTPError(url, status_when_missing, "error", Message(), None)
        trips.to_parquet(dest)

    return download, requested


def test_ingest_trips_only_downloads_months_not_on_disk(
    tmp_path: Path, tlc_trips: pd.DataFrame
) -> None:
    start, end = pd.Period("2026-06"), pd.Period("2026-08")
    download, requested = make_download(tlc_trips, {"2026-06", "2026-07", "2026-08"})

    first = ingest_trips(start, end, data_dir=tmp_path, download=download)
    second = ingest_trips(start, end, data_dir=tmp_path, download=download)

    assert [str(m) for m in first.downloaded] == ["2026-06", "2026-07", "2026-08"]
    assert second.downloaded == []
    assert [str(m) for m in second.already_present] == ["2026-06", "2026-07", "2026-08"]
    assert len(requested) == 3


def test_ingest_trips_skips_months_not_published_yet(
    tmp_path: Path, tlc_trips: pd.DataFrame
) -> None:
    download, _ = make_download(tlc_trips, {"2026-08"})

    result = ingest_trips(
        pd.Period("2026-08"), pd.Period("2026-09"), data_dir=tmp_path, download=download
    )

    assert [str(m) for m in result.downloaded] == ["2026-08"]
    assert [str(m) for m in result.not_published] == ["2026-09"]
    assert [p.name for p in tmp_path.iterdir()] == ["yellow_tripdata_2026-08.parquet"]


def test_ingest_trips_raises_on_other_http_errors(tmp_path: Path, tlc_trips: pd.DataFrame) -> None:
    download, _ = make_download(tlc_trips, set(), status_when_missing=500)

    with pytest.raises(urllib.error.HTTPError):
        ingest_trips(
            pd.Period("2026-08"), pd.Period("2026-08"), data_dir=tmp_path, download=download
        )


def test_ingest_trips_rejects_a_file_missing_expected_columns(
    tmp_path: Path, tlc_trips: pd.DataFrame
) -> None:
    month = pd.Period("2026-08")
    download, _ = make_download(tlc_trips.drop(columns="PULocationID"), {"2026-08"})

    with pytest.raises(ValueError, match="PULocationID"):
        ingest_trips(month, month, data_dir=tmp_path, download=download)

    assert not trip_path(month, tmp_path).exists()
    assert list(tmp_path.iterdir()) == []


def test_ingest_zone_centroids_downloads_and_derives_once(
    tmp_path: Path, taxi_zones_zip: Path
) -> None:
    data_dir = tmp_path / "tlc"
    requested: list[str] = []

    def download(url: str, dest: Path) -> None:
        requested.append(url)
        shutil.copy(taxi_zones_zip, dest)

    first = ingest_zone_centroids(data_dir=data_dir, download=download)
    second = ingest_zone_centroids(data_dir=data_dir, download=download)

    assert first == second == data_dir / CENTROIDS_FILENAME
    assert pd.read_csv(first)["location_id"].tolist() == [1, 2, 3]
    assert len(requested) == 1
