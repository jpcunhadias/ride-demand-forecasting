"""Decide whether the model just trained should replace the one in service.

Both are judged on the same month - the newest one, which the model in service has
never seen - and on the same information:

- the model in service is scored as it is deployed;
- the new model is represented by a copy trained with the new code and settings on
  data up to the same month the model in service was trained to.

Matching the data matters. Demand drifts enough that a model one month behind the test
month beats one two months behind by far more than any sensible margin, so comparing
the new model's ordinary evaluation copy (two months behind) with the model in service
(one month behind) would reject nearly every new model for being older, not worse.

With the data matched, what the comparison measures is the pipeline itself: a change in
code, settings or dependencies that makes the model worse is stopped. It cannot judge
the newest month's data, which no model has been tested beyond; the checks at ingestion
and loading are what guard that.

Promotion moves the `champion` alias in the MLflow model registry, which is what the
API loads at startup. Without an MLflow server there is nothing to promote and the API
keeps serving the model file.
"""

import argparse
import json
import logging
import tempfile
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from ride_demand_forecasting import registry
from ride_demand_forecasting.clustering import assign_pickup_zones
from ride_demand_forecasting.config import (
    MODEL_PATH,
    PROMOTION_PATH,
    PROMOTION_TOLERANCE,
    TLC_DATA_DIR,
)
from ride_demand_forecasting.ingest import CENTROIDS_FILENAME, available_months
from ride_demand_forecasting.train import evaluate, fit, load_months, window_on_disk

logger = logging.getLogger(__name__)


def score_artifact(
    artifact: dict, location_daily: pd.DataFrame, zone_centroids: pd.DataFrame
) -> dict:
    """Score a stored model, as it would serve, on one month of data."""
    points = zone_centroids.rename(columns={"lat": "start_lat", "lng": "start_lng"})
    fitted = {
        **artifact,
        "zones": np.asarray(artifact["zones"]),
        "zone_map": pd.Series(
            assign_pickup_zones(points, artifact["kmeans"]),
            index=points["location_id"].to_numpy(),
        ),
    }
    return evaluate(fitted, location_daily)


def compare(
    candidate: dict, champion: dict, data_dir: Path
) -> tuple[float | None, float | None, str | None]:
    """The new and the current model's errors on the candidate's test month, on equal
    terms - or, when that isn't possible, why not.

    Three cases, by how far the current model's training data reaches:

    - it stops before the test month: score it on that month as deployed, against a
      copy of the new model trained to the same cut-off;
    - it was evaluated on that very month itself (a retrain on the same data): the two
      evaluations are already like for like;
    - anything else means it has seen the test month, so there is no fair comparison.
    """
    test_month = candidate["test_month"]
    last_trained = (champion.get("train_months") or [None])[-1]
    if champion.get("test_month") == test_month and champion.get("metrics"):
        return candidate["metrics"]["wape"], champion["metrics"]["wape"], None
    if last_trained is None:
        return None, None, "the current model doesn't record what it was trained on"
    if last_trained >= test_month:
        obstacle = f"the current model was already trained on data up to {last_trained}"
        return None, None, obstacle

    zone_centroids = pd.read_csv(data_dir / CENTROIDS_FILENAME)
    loaded: dict[pd.Period, pd.DataFrame] = {}
    test_data = load_months([pd.Period(test_month, freq="M")], data_dir, zone_centroids, loaded)
    months = window_on_disk(
        pd.Period(last_trained, freq="M"), candidate["window_months"], available_months(data_dir)
    )
    if not months:
        return None, None, f"no data on disk up to {last_trained} to train a matching copy on"
    matched_copy = fit(
        load_months(months, data_dir, zone_centroids, loaded),
        zone_centroids,
        len(candidate["zones"]),
    )
    return (
        evaluate(matched_copy, test_data)["wape"],
        score_artifact(champion, test_data, zone_centroids)["wape"],
        None,
    )


def decide(
    candidate_wape: float | None, current_wape: float | None, tolerance: float
) -> tuple[bool, str]:
    """The promotion rule on its own: (promote?, why)."""
    if candidate_wape is None:
        return False, "the new model was not evaluated, so it can't be compared"
    if current_wape is None:
        return True, "there is no current model"
    limit = current_wape * (1 + tolerance)
    if candidate_wape <= limit:
        return True, (
            f"WAPE {candidate_wape:.4f} is within the limit of {limit:.4f} "
            f"(current model {current_wape:.4f})"
        )
    return False, (
        f"WAPE {candidate_wape:.4f} is above the limit of {limit:.4f} "
        f"(current model {current_wape:.4f})"
    )


def save_decision(decision: dict, path: str | Path = PROMOTION_PATH) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(decision, indent=2) + "\n")


def promote(
    model_path: str | Path = MODEL_PATH,
    data_dir: str | Path = TLC_DATA_DIR,
    tolerance: float = PROMOTION_TOLERANCE,
) -> dict:
    """Register the model at `model_path` and promote it if the rule allows."""
    decision: dict[str, Any] = {
        "promoted": False,
        "reason": "MLflow is not configured, so there is no registry to promote in",
        "candidate_version": None,
        "previous_version": None,
        # The two figures the rule compared, and the new model's own evaluation score
        # (trained two months before the test month), which is the one to quote.
        "candidate_wape": None,
        "current_wape": None,
        "evaluation_wape": None,
        "test_month": None,
    }
    if not registry.tracking_enabled():
        return decision

    from mlflow import MlflowClient

    candidate = joblib.load(model_path)
    run = registry.find_training_run(candidate["trained_at"])
    if run is None:
        raise RuntimeError(
            f"{model_path} was not logged to MLflow. Run `ride-demand-train` with "
            "MLFLOW_TRACKING_URI set, then promote."
        )
    version = registry.register_run(run.info.run_id)
    current = registry.champion()
    candidate_wape = candidate["metrics"].get("wape")
    decision.update(
        candidate_version=int(version.version),
        previous_version=int(current.version) if current else None,
        candidate_wape=candidate_wape,
        evaluation_wape=candidate_wape,
        test_month=candidate["test_month"],
    )

    if current is not None and current.run_id == run.info.run_id:
        # Nothing to decide, and nothing to record over the decision that promoted it.
        decision["reason"] = "this model is already the current one"
        return decision
    if current is not None and candidate_wape is not None:
        with tempfile.TemporaryDirectory() as tmp:
            champion = joblib.load(registry.download_model(current, tmp))
        matched_wape, current_wape, obstacle = compare(candidate, champion, Path(data_dir))
        decision.update(candidate_wape=matched_wape, current_wape=current_wape)
        if obstacle is not None:
            decision["reason"] = f"no fair comparison: {obstacle}"
        else:
            decision["promoted"], decision["reason"] = decide(matched_wape, current_wape, tolerance)
    else:
        decision["promoted"], decision["reason"] = decide(candidate_wape, None, tolerance)

    client = MlflowClient()
    if decision["promoted"]:
        registry.set_champion(version.version)
    client.set_tag(run.info.run_id, "promoted", str(decision["promoted"]).lower())
    client.set_tag(run.info.run_id, "promotion_reason", decision["reason"])
    if decision["current_wape"] is not None:
        client.log_metric(run.info.run_id, "current_model_wape", decision["current_wape"])
    return decision


def describe_versions() -> pd.DataFrame:
    """One row per registered version: what it was trained on, its score, and which
    one is in service."""
    from mlflow import MlflowClient

    client = MlflowClient()
    current = registry.champion()
    rows = []
    for version in registry.list_versions():
        run = client.get_run(version.run_id)
        rows.append(
            {
                "version": int(version.version),
                "current": "*" if current and current.version == version.version else "",
                "trained_to": run.data.params.get("train_last_month"),
                "test_month": run.data.params.get("test_month"),
                "wape": run.data.metrics.get("wape"),
                "promoted": run.data.tags.get("promoted", ""),
            }
        )
    return pd.DataFrame(
        rows, columns=["version", "current", "trained_to", "test_month", "wape", "promoted"]
    )


def main() -> None:
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    parser = argparse.ArgumentParser(
        description="Promote the latest trained model if it passes, or manage versions."
    )
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "--version",
        type=int,
        help="put this version in service regardless of the rule (rollback or override)",
    )
    group.add_argument("--list", action="store_true", help="show the registered versions")
    args = parser.parse_args()

    if (args.version is not None or args.list) and not registry.tracking_enabled():
        raise SystemExit("MLFLOW_TRACKING_URI is not set, so there is no registry to use.")

    if args.list:
        print(describe_versions().to_string(index=False, float_format="%.4f"))
        return
    if args.version is not None:
        previous = registry.champion()
        registry.set_champion(args.version)
        logger.info(
            "Version %d is now the current model (was %s). Restart the API to serve it.",
            args.version,
            previous.version if previous else "none",
        )
        return

    decision = promote()
    save_decision(decision)
    logger.info(
        "%s: %s", "Promoted" if decision["promoted"] else "Not promoted", decision["reason"]
    )
    if decision["promoted"]:
        logger.info(
            "Version %d is now the current model. Restart the API to serve it.",
            decision["candidate_version"],
        )


if __name__ == "__main__":
    main()
