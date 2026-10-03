"""Checks that justify two pipeline settings with measurements rather than guesses:
how many months to train on, and how many pickup zones to cluster into.

Neither changes anything on its own - they print a comparison, and the chosen values
are then set in `config.py`.
"""

import argparse
import logging
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pandas as pd

from ride_demand_forecasting.clustering import fit_pickup_zones, weighted_silhouette
from ride_demand_forecasting.config import (
    N_PICKUP_ZONES,
    PICKUP_ZONE_CANDIDATES,
    TLC_BACKTEST_TEST_MONTHS,
    TLC_BACKTEST_WINDOWS,
    TLC_DATA_DIR,
    TLC_EVAL_GAP_MONTHS,
    TLC_TRAIN_WINDOW_MONTHS,
)
from ride_demand_forecasting.ingest import CENTROIDS_FILENAME, available_months
from ride_demand_forecasting.tracking import log_backtest
from ride_demand_forecasting.train import evaluate, fit, load_months, pickup_points

logger = logging.getLogger(__name__)

LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"


def backtest(
    data_dir: str | Path = TLC_DATA_DIR,
    windows: Sequence[int] = TLC_BACKTEST_WINDOWS,
    zone_counts: Sequence[int] = (N_PICKUP_ZONES,),
    n_test_months: int = TLC_BACKTEST_TEST_MONTHS,
    gap_months: int = TLC_EVAL_GAP_MONTHS,
) -> pd.DataFrame:
    """Score each training-window length and pickup-zone count on each of the most
    recent test months.

    For a test month, a window of `w` months ends `gap_months` before it - the same
    distance the publication lag puts between a model and what it predicts. A window
    that isn't fully on disk for a test month is skipped rather than run short, since a
    short window would be scored under the wrong label.

    Zone counts are meant to be compared on `wape`: `mae` and `rmse` shrink as zones
    get smaller whether or not the forecast improves. `smallest_zone_share` is the
    quietest zone's share of the training window's pickups.

    Returns one row per (window, zone count, test month) that could be run.
    """
    data_dir = Path(data_dir)
    available = available_months(data_dir)
    on_disk = set(available)
    zone_centroids = pd.read_csv(data_dir / CENTROIDS_FILENAME)
    loaded: dict[pd.Period, pd.DataFrame] = {}

    rows = []
    for test_month in available[-n_test_months:]:
        for window in windows:
            train_months = list(
                pd.period_range(end=test_month - gap_months, periods=window, freq="M")
            )
            if not on_disk.issuperset(train_months):
                logger.info(
                    "Skipping the %d-month window for %s: not all of it is on disk",
                    window,
                    test_month,
                )
                continue
            train_data = load_months(train_months, data_dir, zone_centroids, loaded)
            test_data = load_months([test_month], data_dir, zone_centroids, loaded)
            n_pickup_points = len(pickup_points(train_data, zone_centroids)[0])
            for n_zones in zone_counts:
                if n_zones >= n_pickup_points:
                    logger.info(
                        "Skipping %d zones for %s: only %d taxi zones had pickups",
                        n_zones,
                        test_month,
                        n_pickup_points,
                    )
                    continue
                fitted = fit(train_data, zone_centroids, n_zones)
                metrics = evaluate(fitted, test_data)
                zone_rides = train_data.groupby(
                    train_data["pickup_location_id"].map(fitted["zone_map"])
                )["ride_count"].sum()
                smallest_zone_share = float(
                    zone_rides.reindex(fitted["zones"], fill_value=0).min() / zone_rides.sum()
                )
                logger.info(
                    "window=%d zones=%d test=%s WAPE=%.1f%%",
                    window,
                    n_zones,
                    test_month,
                    100 * metrics["wape"],
                )
                rows.append(
                    {
                        "window_months": window,
                        "n_zones": n_zones,
                        "test_month": str(test_month),
                        **metrics,
                        "smallest_zone_share": smallest_zone_share,
                    }
                )
    return pd.DataFrame(
        rows,
        columns=[
            "window_months",
            "n_zones",
            "test_month",
            "mae",
            "rmse",
            "wape",
            "smallest_zone_share",
            "n_test_rows",
        ],
    )


def summarize_backtest(results: pd.DataFrame) -> pd.DataFrame:
    """Average each setting's scores over the test months every setting was run on, so
    the settings are compared on exactly the same months.

    If no test month was run for every setting, all of each setting's months are used
    instead and the differing `test_months` counts show the comparison is uneven.
    """
    setting = ["window_months", "n_zones"]
    n_settings = len(results[setting].drop_duplicates())
    runs_per_month = results.groupby("test_month").size()
    common = runs_per_month[runs_per_month == n_settings].index
    if len(common) > 0:
        results = results[results["test_month"].isin(common)]
    return (
        results.groupby(setting)
        .agg(
            test_months=("test_month", "nunique"),
            mae=("mae", "mean"),
            rmse=("rmse", "mean"),
            wape=("wape", "mean"),
            smallest_zone_share=("smallest_zone_share", "mean"),
        )
        .reset_index()
    )


def validate_k(
    data_dir: str | Path = TLC_DATA_DIR,
    k_values: Sequence[int] = PICKUP_ZONE_CANDIDATES,
    window_months: int = TLC_TRAIN_WINDOW_MONTHS,
) -> pd.DataFrame:
    """Cluster the latest training window's pickups into each candidate number of zones.

    Returns one row per `k` with the usual two clustering scores, both weighted by
    pickups, plus how small the quietest zone gets - a zone with a sliver of the demand
    gives the model almost nothing to learn from.
    """
    data_dir = Path(data_dir)
    available = available_months(data_dir)
    zone_centroids = pd.read_csv(data_dir / CENTROIDS_FILENAME)
    location_daily = load_months(available[-window_months:], data_dir, zone_centroids, {})
    points, weights = pickup_points(location_daily, zone_centroids)
    coords = points[["start_lat", "start_lng"]].to_numpy()

    rows = []
    for k in k_values:
        if k >= len(points):
            logger.info("Skipping k=%d: only %d zones had pickups", k, len(points))
            continue
        kmeans = fit_pickup_zones(points, n_zones=k, sample_weight=weights)
        zone_share = np.bincount(kmeans.labels_, weights=weights, minlength=k) / weights.sum()
        rows.append(
            {
                "k": k,
                "inertia": float(kmeans.inertia_),
                "silhouette": weighted_silhouette(coords, kmeans.labels_, weights),
                "smallest_zone_share": float(zone_share.min()),
                "largest_zone_share": float(zone_share.max()),
            }
        )
    return pd.DataFrame(
        rows, columns=["k", "inertia", "silhouette", "smallest_zone_share", "largest_zone_share"]
    )


def backtest_main() -> None:
    logging.basicConfig(level=logging.INFO, format=LOG_FORMAT)
    parser = argparse.ArgumentParser(
        description="Compare training-window lengths and pickup-zone counts on recent test months."
    )
    parser.add_argument(
        "--windows",
        type=int,
        nargs="+",
        default=list(TLC_BACKTEST_WINDOWS),
        help="window lengths in months (default: %(default)s)",
    )
    parser.add_argument(
        "--zones",
        type=int,
        nargs="+",
        default=[N_PICKUP_ZONES],
        help="numbers of pickup zones to compare, judged on WAPE (default: %(default)s)",
    )
    parser.add_argument(
        "--test-months",
        type=int,
        default=TLC_BACKTEST_TEST_MONTHS,
        help="how many of the most recent months to test on (default: %(default)s)",
    )
    args = parser.parse_args()

    results = backtest(windows=args.windows, zone_counts=args.zones, n_test_months=args.test_months)
    if results.empty:
        raise SystemExit(
            "Nothing to compare: no test month has a full training window before it on disk. "
            "Ingest more months with `ride-demand-ingest --start YYYY-MM`."
        )
    summary = summarize_backtest(results)
    print(results.to_string(index=False, float_format="%.4f"))
    print()
    print(summary.to_string(index=False, float_format="%.4f"))
    log_backtest(summary, TLC_EVAL_GAP_MONTHS)


def validate_k_main() -> None:
    logging.basicConfig(level=logging.INFO, format=LOG_FORMAT)
    parser = argparse.ArgumentParser(
        description="Compare candidate numbers of pickup zones on the latest training window."
    )
    parser.add_argument(
        "--k",
        type=int,
        nargs="+",
        default=list(PICKUP_ZONE_CANDIDATES),
        help="numbers of zones to try (default: %(default)s)",
    )
    args = parser.parse_args()

    results = validate_k(k_values=args.k)
    print(results.to_string(index=False, float_format="%.4f"))
    if not results.empty:
        best = results.loc[results["silhouette"].idxmax()]
        print(f"\nHighest silhouette: k={int(best['k'])} ({best['silhouette']:.4f})")
