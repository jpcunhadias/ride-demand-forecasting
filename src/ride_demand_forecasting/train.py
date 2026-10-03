"""End-to-end training pipeline: ingested TLC trips -> persisted model artifact.

Ported from notebooks/ride_demand_forecasting.ipynb's baseline-modeling and
Model Validation sections, with the changes that matter for a real serving
pipeline fed by monthly data (the notebook itself is left as the exploratory
record on the 2016 dataset, unchanged):

1. Training uses a rolling window of the most recent months rather than one
   fixed file, so each run reflects current demand.
2. The model is evaluated the way it is used. Trip data arrives about two
   months late, so the evaluation model is trained on a window ending two
   months before the test month. The model that is saved is then refit on the
   window ending at the latest month - the reported metrics describe the
   procedure, not that exact model.
3. The pickup-zone KMeans is fit on the training window only, over TLC zone
   centre points weighted by how many pickups each zone had.
4. `avg_ride_duration_min` / `avg_ride_distance` aren't accepted as inputs to
   the serving API (a caller can't know them ahead of a future prediction
   window) - a historical (pickup_zone, hour) profile is built from training
   data instead, and stored in the artifact for lookup.
5. Training rows are day-level (one `(pickup_zone, hour, date)` observation
   each), zero-filled so the model sees genuine zero-demand zone-hour-days,
   and carry the day of the week - the one input that varies within a
   zone-hour, and so the one that lets the model beat a plain average.
6. LightGBM (tuned) replaces the untuned XGBoost baseline - it edged out an
   identically-tuned XGBoost in the notebook's Model Validation comparison.
"""

import argparse
import json
import logging
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import OrdinalEncoder

from ride_demand_forecasting.clustering import assign_pickup_zones, fit_pickup_zones
from ride_demand_forecasting.config import (
    FEATURE_ORDER,
    LGBM_PARAMS,
    METRICS_PATH,
    MODEL_PATH,
    N_PICKUP_ZONES,
    RANDOM_STATE,
    TLC_DATA_DIR,
    TLC_EVAL_GAP_MONTHS,
    TLC_TRAIN_WINDOW_MONTHS,
)
from ride_demand_forecasting.data import load_tlc_trips, validate_daily_coverage
from ride_demand_forecasting.features import (
    aggregate_location_hour_daily,
    zone_hour_daily_counts,
    zone_hour_profile,
)
from ride_demand_forecasting.ingest import CENTROIDS_FILENAME, available_months, trip_path
from ride_demand_forecasting.tracking import log_training_run

logger = logging.getLogger(__name__)


def load_month(month: pd.Period, data_dir: Path, zone_centroids: pd.DataFrame) -> pd.DataFrame:
    """One ingested month, cleaned and reduced to (location, date, hour) rows."""
    trips = load_tlc_trips(trip_path(month, data_dir), month, zone_centroids)
    location_daily = aggregate_location_hour_daily(trips)
    validate_daily_coverage(location_daily, month)
    return location_daily


def load_months(
    months: list[pd.Period],
    data_dir: Path,
    zone_centroids: pd.DataFrame,
    loaded: dict[pd.Period, pd.DataFrame],
) -> pd.DataFrame:
    """Stack the given months, reading each from disk at most once across calls that
    share the same `loaded` cache."""
    for month in months:
        if month not in loaded:
            loaded[month] = load_month(month, data_dir, zone_centroids)
    return pd.concat([loaded[month] for month in months], ignore_index=True)


def window_on_disk(
    last_month: pd.Period, n_months: int, available: list[pd.Period]
) -> list[pd.Period]:
    """The months of the `n_months` calendar window ending at `last_month` that are on
    disk. A missing month shortens the window; it is never replaced by an older one."""
    wanted = pd.period_range(end=last_month, periods=n_months, freq="M")
    on_disk = set(available)
    present = [month for month in wanted if month in on_disk]
    if present and len(present) < len(wanted):
        missing = ", ".join(str(month) for month in wanted if month not in on_disk)
        logger.warning("Window ending %s is missing: %s", last_month, missing)
    return present


def pickup_points(
    location_daily: pd.DataFrame, zone_centroids: pd.DataFrame
) -> tuple[pd.DataFrame, np.ndarray]:
    """The centre point of every zone that had pickups, and how many it had."""
    points = zone_centroids.rename(columns={"lat": "start_lat", "lng": "start_lng"})
    pickups = location_daily.groupby("pickup_location_id")["ride_count"].sum()
    seen = points[points["location_id"].isin(pickups.index)]
    return seen, pickups.loc[seen["location_id"]].to_numpy()


def features_and_target(
    location_daily: pd.DataFrame, fitted: dict
) -> tuple[pd.DataFrame, pd.Series]:
    daily = zone_hour_daily_counts(location_daily, fitted["zone_map"], fitted["zones"]).merge(
        fitted["zone_hour_profile"], on=["pickup_zone", "hour"], how="left"
    )
    daily["day_of_week"] = daily["date"].dt.dayofweek
    X = daily[FEATURE_ORDER].copy()
    X[["pickup_zone"]] = fitted["encoder"].transform(X[["pickup_zone"]])
    return X, daily["ride_count"]


def fit_zones(location_daily: pd.DataFrame, zone_centroids: pd.DataFrame, n_zones: int) -> dict:
    """Fit everything that comes before the model on one training window: the pickup
    zones and the zone-hour profile."""
    # Zones with no pickups in the window don't shape the clusters, but still get
    # assigned to one so any location can be mapped at serving time.
    seen, weights = pickup_points(location_daily, zone_centroids)
    kmeans = fit_pickup_zones(seen, n_zones=n_zones, sample_weight=weights)
    points = zone_centroids.rename(columns={"lat": "start_lat", "lng": "start_lng"})
    zones = np.arange(n_zones)
    fitted = {
        "kmeans": kmeans,
        "zone_map": pd.Series(
            assign_pickup_zones(points, kmeans), index=points["location_id"].to_numpy()
        ),
        "zones": zones,
        "encoder": OrdinalEncoder().fit(pd.DataFrame({"pickup_zone": zones})),
    }
    fitted["zone_hour_profile"] = zone_hour_profile(location_daily, fitted["zone_map"], zones)
    return fitted


def fit_model(X_train: pd.DataFrame, y_train: pd.Series, lgbm_params: dict) -> LGBMRegressor:
    return LGBMRegressor(**lgbm_params, random_state=RANDOM_STATE, verbose=-1).fit(X_train, y_train)


def fit(location_daily: pd.DataFrame, zone_centroids: pd.DataFrame, n_zones: int) -> dict:
    """Fit pickup zones, the zone-hour profile and the model on one training window."""
    fitted = fit_zones(location_daily, zone_centroids, n_zones)
    X_train, y_train = features_and_target(location_daily, fitted)
    fitted["model"] = fit_model(X_train, y_train, LGBM_PARAMS)
    fitted["n_train_rows"] = len(X_train)
    return fitted


def score(y_true: pd.Series, y_pred: np.ndarray) -> dict:
    return {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        # Total absolute error as a share of total rides. Unlike MAE it doesn't depend
        # on how busy a zone-hour typically is, so it compares across zone layouts.
        "wape": float(np.abs(y_true - y_pred).sum() / y_true.sum()),
    }


def evaluate(fitted: dict, location_daily: pd.DataFrame) -> dict:
    X_test, y_test = features_and_target(location_daily, fitted)
    return {**score(y_test, fitted["model"].predict(X_test)), "n_test_rows": len(X_test)}


def train(
    data_dir: str | Path = TLC_DATA_DIR,
    end_month: pd.Period | None = None,
    window_months: int = TLC_TRAIN_WINDOW_MONTHS,
    gap_months: int = TLC_EVAL_GAP_MONTHS,
    n_zones: int = N_PICKUP_ZONES,
) -> dict:
    """Run the full pipeline and return the artifact dict (unsaved).

    `end_month` is the last month of training data, and also the month the evaluation
    model is tested on; it defaults to the latest month on disk.
    """
    data_dir = Path(data_dir)
    available = available_months(data_dir)
    if not available:
        raise FileNotFoundError(f"No TLC trip files in {data_dir} - run `ride-demand-ingest`")
    end_month = end_month or available[-1]
    if end_month not in available:
        raise FileNotFoundError(f"No TLC trip file for {end_month} in {data_dir}")
    zone_centroids = pd.read_csv(data_dir / CENTROIDS_FILENAME)

    loaded: dict[pd.Period, pd.DataFrame] = {}

    def load_window(last_month: pd.Period) -> tuple[pd.DataFrame | None, list[pd.Period]]:
        present = window_on_disk(last_month, window_months, available)
        if not present:
            return None, []
        return load_months(present, data_dir, zone_centroids, loaded), present

    train_data, train_months = load_window(end_month)
    assert train_data is not None  # `end_month` itself is always present

    eval_data, eval_months = load_window(end_month - gap_months)
    if eval_data is not None:
        metrics = evaluate(fit(eval_data, zone_centroids, n_zones), loaded[end_month])
    else:
        logger.warning(
            "No data %d months before %s to train an evaluation model on - skipping evaluation",
            gap_months,
            end_month,
        )
        metrics = {}
    n_test_rows = metrics.pop("n_test_rows", 0)

    fitted = fit(train_data, zone_centroids, n_zones)
    return {
        "kmeans": fitted["kmeans"],
        "encoder": fitted["encoder"],
        "model": fitted["model"],
        "zone_hour_profile": fitted["zone_hour_profile"],
        "zone_map": fitted["zone_map"].to_dict(),
        "zones": fitted["zones"].tolist(),
        "feature_order": FEATURE_ORDER,
        "metrics": metrics,
        "trained_at": pd.Timestamp.now("UTC").isoformat(),
        "train_months": [str(month) for month in train_months],
        "eval_train_months": [str(month) for month in eval_months],
        "test_month": str(end_month) if eval_months else None,
        "window_months": window_months,
        "gap_months": gap_months,
        "n_train_rows": fitted["n_train_rows"],
        "n_test_rows": n_test_rows,
    }


def save_artifact(artifact: dict, model_path: str | Path = MODEL_PATH) -> None:
    model_path = Path(model_path)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(artifact, model_path)


def save_metrics(artifact: dict, metrics_path: str | Path = METRICS_PATH) -> None:
    metrics_path = Path(metrics_path)
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    metrics = {
        **artifact["metrics"],
        "n_train_rows": artifact["n_train_rows"],
        "n_test_rows": artifact["n_test_rows"],
    }
    metrics_path.write_text(json.dumps(metrics, indent=2) + "\n")


def main() -> None:
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    parser = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
    parser.add_argument(
        "--end-month",
        type=pd.Period,
        default=None,
        help="last month of training data, as YYYY-MM (default: the latest month on disk)",
    )
    args = parser.parse_args()

    start = time.perf_counter()
    logger.info("Training on %s ...", TLC_DATA_DIR)
    artifact = train(end_month=args.end_month)
    save_artifact(artifact)
    save_metrics(artifact)
    elapsed = time.perf_counter() - start
    logger.info("Saved model artifact to %s in %.1fs", MODEL_PATH, elapsed)
    logger.info("Trained on %s to %s", artifact["train_months"][0], artifact["train_months"][-1])
    if artifact["metrics"]:
        logger.info(
            "Evaluated on %s with a model trained up to %s",
            artifact["test_month"],
            artifact["eval_train_months"][-1],
        )
        logger.info("Test MAE: %.3f rides/day", artifact["metrics"]["mae"])
        logger.info("Test RMSE: %.3f rides/day", artifact["metrics"]["rmse"])
        logger.info("Test WAPE: %.1f%% of rides", 100 * artifact["metrics"]["wape"])
    log_training_run(artifact, MODEL_PATH)


if __name__ == "__main__":
    main()
