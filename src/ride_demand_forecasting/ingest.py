"""Incremental download of NYC TLC yellow taxi trip records.

The TLC publishes one Parquet file per month, roughly two months in arrears. Each run
fetches only the months that aren't already on disk, so re-running is cheap and safe:
a month that's already present is left alone, and one that isn't published yet is
skipped without failing - with a two-month lag, "nothing new" is a normal outcome.

Files are stored exactly as published. Cleaning and the zone centre-point join happen
at load time (see `data.load_tlc_trips`).
"""

import argparse
import logging
import os
import shutil
import urllib.error
import urllib.request
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from ride_demand_forecasting.config import (
    TLC_BASE_URL,
    TLC_DATA_DIR,
    TLC_DEFAULT_LOOKBACK_MONTHS,
)
from ride_demand_forecasting.zones import compute_zone_centroids

logger = logging.getLogger(__name__)

REQUIRED_TRIP_COLUMNS = {
    "tpep_pickup_datetime",
    "tpep_dropoff_datetime",
    "PULocationID",
    "DOLocationID",
    "trip_distance",
    "fare_amount",
}
ZONES_FILENAME = "taxi_zones.zip"
CENTROIDS_FILENAME = "taxi_zone_centroids.csv"
# The CDN answers 403, not 404, for a month that hasn't been published.
NOT_PUBLISHED_STATUS = {403, 404}
DOWNLOAD_TIMEOUT_S = 60

Download = Callable[[str, Path], None]


@dataclass
class IngestResult:
    downloaded: list[pd.Period] = field(default_factory=list)
    already_present: list[pd.Period] = field(default_factory=list)
    not_published: list[pd.Period] = field(default_factory=list)


def trip_path(month: pd.Period, data_dir: str | Path = TLC_DATA_DIR) -> Path:
    return Path(data_dir) / f"yellow_tripdata_{month}.parquet"


def trip_url(month: pd.Period) -> str:
    return f"{TLC_BASE_URL}/trip-data/yellow_tripdata_{month}.parquet"


def download_file(url: str, dest: Path) -> None:
    with (
        urllib.request.urlopen(url, timeout=DOWNLOAD_TIMEOUT_S) as response,
        dest.open("wb") as out,
    ):
        shutil.copyfileobj(response, out)


def _fetch(url: str, dest: Path, download: Download) -> Path:
    """Download to a temporary sibling of `dest`, so a failed or interrupted download
    never leaves a partial file where a complete one is expected."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    # Keeps the extension, which is how the shapefile reader recognises a zip.
    partial = dest.with_name(f"partial-{dest.name}")
    try:
        download(url, partial)
    except BaseException:
        partial.unlink(missing_ok=True)
        raise
    return partial


def validate_trip_file(path: Path) -> None:
    """Fail early if the file can't be read in full, has lost a column the pipeline
    relies on, or has one in an unexpected type.

    Every column is decoded, not just listed: a file's schema lives in its footer and
    can be intact while the data itself is damaged. That includes columns the pipeline
    doesn't use, since the file is kept as the record of what was published.
    """
    try:
        parquet = pq.ParquetFile(path)
        schema = parquet.schema_arrow
        missing = REQUIRED_TRIP_COLUMNS - set(schema.names)
        if missing:
            raise ValueError(f"{path.name} is missing expected columns: {sorted(missing)}")
        for column in sorted(REQUIRED_TRIP_COLUMNS):
            column_type = schema.field(column).type
            is_timestamp = pa.types.is_timestamp(column_type)
            is_number = pa.types.is_integer(column_type) or pa.types.is_floating(column_type)
            if is_timestamp != column.endswith("_datetime") or not (is_timestamp or is_number):
                raise ValueError(f"{path.name} has an unexpected type for {column}: {column_type}")
        for _ in parquet.iter_batches():
            pass
    except (OSError, pa.ArrowException) as error:
        raise ValueError(f"{path.name} could not be read: {error}") from error


def ingest_trips(
    start: pd.Period,
    end: pd.Period,
    data_dir: str | Path = TLC_DATA_DIR,
    download: Download = download_file,
) -> IngestResult:
    """Fetch every month in `start`..`end` (inclusive) that isn't already on disk."""
    result = IngestResult()
    for month in pd.period_range(start, end, freq="M"):
        dest = trip_path(month, data_dir)
        if dest.exists():
            result.already_present.append(month)
            continue
        try:
            partial = _fetch(trip_url(month), dest, download)
        except urllib.error.HTTPError as error:
            if error.code in NOT_PUBLISHED_STATUS:
                logger.info("%s is not published yet", month)
                result.not_published.append(month)
                continue
            raise
        try:
            validate_trip_file(partial)
        except BaseException:
            partial.unlink(missing_ok=True)
            raise
        os.replace(partial, dest)
        logger.info("Downloaded %s (%.1f MB)", dest.name, dest.stat().st_size / 1e6)
        result.downloaded.append(month)
    return result


def ingest_zone_centroids(
    data_dir: str | Path = TLC_DATA_DIR, download: Download = download_file
) -> Path:
    """Fetch the taxi zone boundary file and derive the centre-point lookup from it,
    unless the lookup is already on disk."""
    data_dir = Path(data_dir)
    centroids_path = data_dir / CENTROIDS_FILENAME
    if centroids_path.exists():
        return centroids_path

    # Nothing is put at its final path until it is known to be good: a boundary file
    # only after centre points have been derived from it, and the lookup only once it
    # is fully written. A run that fails part-way leaves nothing for the next to trust.
    zones_zip = data_dir / ZONES_FILENAME
    centroids = None
    if zones_zip.exists():
        try:
            centroids = compute_zone_centroids(zones_zip)
        except Exception:
            logger.warning("%s can't be read - downloading it again", zones_zip)
            zones_zip.unlink()
    if centroids is None:
        downloaded = _fetch(f"{TLC_BASE_URL}/misc/{ZONES_FILENAME}", zones_zip, download)
        try:
            centroids = compute_zone_centroids(downloaded)
        except BaseException:
            downloaded.unlink(missing_ok=True)
            raise
        os.replace(downloaded, zones_zip)

    partial_lookup = centroids_path.with_name(f"partial-{centroids_path.name}")
    centroids.to_csv(partial_lookup, index=False)
    os.replace(partial_lookup, centroids_path)
    logger.info("Wrote %d zone centre points to %s", len(centroids), centroids_path)
    return centroids_path


def main() -> None:
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    current_month = pd.Timestamp.now().to_period("M")
    parser = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
    parser.add_argument(
        "--start",
        type=pd.Period,
        default=current_month - TLC_DEFAULT_LOOKBACK_MONTHS,
        help="first month to fetch, as YYYY-MM (default: %(default)s)",
    )
    parser.add_argument(
        "--end",
        type=pd.Period,
        default=current_month,
        help="last month to fetch, as YYYY-MM (default: the current month)",
    )
    args = parser.parse_args()

    ingest_zone_centroids()
    result = ingest_trips(args.start, args.end)
    logger.info(
        "%d month(s) downloaded, %d already present, %d not published yet",
        len(result.downloaded),
        len(result.already_present),
        len(result.not_published),
    )


if __name__ == "__main__":
    main()
