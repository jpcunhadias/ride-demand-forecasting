"""Centre points for the NYC TLC taxi zones.

TLC trip records identify pickups and dropoffs by taxi zone ID rather than by
coordinates. Replacing each ID with its zone's centre point gives the rest of the
pipeline the same `start_lat`/`start_lng` shape it was built around, so pickup zones
can still be clustered with KMeans (see `clustering.py`).
"""

import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import shapefile
from pyproj import CRS, Transformer

CENTROID_COLUMNS = ["location_id", "zone", "borough", "lat", "lng"]


def _ring_moments(points: np.ndarray) -> tuple[float, float, float]:
    """Signed area and first moments of one closed polygon ring (shoelace formula)."""
    x, y = points[:, 0], points[:, 1]
    cross = x[:-1] * y[1:] - x[1:] * y[:-1]
    area = cross.sum() / 2
    moment_x = ((x[:-1] + x[1:]) * cross).sum() / 6
    moment_y = ((y[:-1] + y[1:]) * cross).sum() / 6
    return area, moment_x, moment_y


def _shape_moments(shape: shapefile.Shape) -> tuple[float, float, float]:
    """Summed signed area and moments over every ring of a shape.

    Shapefile holes wind the opposite way to outer rings, so summing signed values
    subtracts them without having to tell the two apart.
    """
    points = np.asarray(shape.points, dtype=float)
    # Coordinates are in the millions (state-plane feet); working relative to the first
    # point keeps the cross products small enough to stay precise.
    origin = points[0]
    points = points - origin
    ring_starts = [*shape.parts, len(points)]

    area = moment_x = moment_y = 0.0
    for start, end in zip(ring_starts[:-1], ring_starts[1:], strict=True):
        ring_area, ring_mx, ring_my = _ring_moments(points[start:end])
        area += ring_area
        moment_x += ring_mx
        moment_y += ring_my
    # Shift the moments back to absolute coordinates.
    return area, moment_x + area * origin[0], moment_y + area * origin[1]


def compute_zone_centroids(zones_zip: str | Path) -> pd.DataFrame:
    """Area-weighted centre point of every taxi zone in the TLC's zipped shapefile.

    Returns one row per `location_id` with `lat`/`lng` in WGS84. Shapes that share a
    location ID (some versions of the file have had a few) are merged into a single
    centre point, so the ID is always a unique join key.
    """
    zones_zip = Path(zones_zip)
    with zipfile.ZipFile(zones_zip) as archive:
        prj_name = next(name for name in archive.namelist() if name.lower().endswith(".prj"))
        source_crs = CRS.from_wkt(archive.read(prj_name).decode())
    to_wgs84 = Transformer.from_crs(source_crs, "EPSG:4326", always_xy=True)

    rows = []
    with shapefile.Reader(str(zones_zip)) as reader:
        for shape_record in reader.iterShapeRecords():
            record = shape_record.record.as_dict()
            area, moment_x, moment_y = _shape_moments(shape_record.shape)
            rows.append(
                {
                    "location_id": int(record["LocationID"]),
                    "zone": record["zone"],
                    "borough": record["borough"],
                    "area": area,
                    "moment_x": moment_x,
                    "moment_y": moment_y,
                }
            )

    merged = (
        pd.DataFrame(rows)
        .groupby("location_id", as_index=False)
        .agg(
            zone=("zone", "first"),
            borough=("borough", "first"),
            area=("area", "sum"),
            moment_x=("moment_x", "sum"),
            moment_y=("moment_y", "sum"),
        )
        .sort_values("location_id")
    )
    lng, lat = to_wgs84.transform(
        (merged["moment_x"] / merged["area"]).to_numpy(),
        (merged["moment_y"] / merged["area"]).to_numpy(),
    )
    merged["lat"] = lat
    merged["lng"] = lng
    return merged[CENTROID_COLUMNS].reset_index(drop=True)
