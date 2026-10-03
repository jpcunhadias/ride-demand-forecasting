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
import io
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

ALREADY_CURRENT = "this model is already the current one"


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


def _month(value: str | None) -> pd.Period | None:
    return None if value is None else pd.Period(value, freq="M")


def _last_month(months: list[str] | None) -> pd.Period | None:
    return _month(months[-1]) if months else None


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
    test_month = pd.Period(candidate["test_month"], freq="M")
    trained_months = champion.get("train_months") or []
    if not trained_months:
        return None, None, "the current model doesn't record what it was trained on"
    cutoff = pd.Period(trained_months[-1], freq="M")

    # The stored evaluations are like for like only if both were tested on the same
    # month and both are known to have stopped training at the same earlier month.
    if (
        champion.get("metrics")
        and _month(champion.get("test_month")) == test_month
        and _last_month(candidate.get("eval_train_months")) is not None
        and _last_month(champion.get("eval_train_months"))
        == _last_month(candidate.get("eval_train_months"))
    ):
        return candidate["metrics"]["wape"], champion["metrics"]["wape"], None
    if cutoff >= test_month:
        return None, None, f"the current model was already trained on data up to {cutoff}"

    zone_centroids = pd.read_csv(data_dir / CENTROIDS_FILENAME)
    loaded: dict[pd.Period, pd.DataFrame] = {}
    months = window_on_disk(cutoff, candidate["window_months"], available_months(data_dir))
    if not months or months[-1] != cutoff:
        # A copy that stops earlier than the current model would be compared at a
        # disadvantage, which is the very thing matching the data is meant to avoid.
        return None, None, f"the data for {cutoff} is not on disk to train a matching copy on"
    test_data = load_months([test_month], data_dir, zone_centroids, loaded)
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

    # Read once: the same bytes are checked against the logged digest and loaded.
    content = Path(model_path).read_bytes()
    candidate = joblib.load(io.BytesIO(content))
    run = registry.find_training_run(candidate["trained_at"])
    if run is None:
        raise RuntimeError(
            f"{model_path} was not logged to MLflow. Run `ride-demand-train` with "
            "MLFLOW_TRACKING_URI set, then promote."
        )
    if run.data.tags.get("model_sha256") != registry.sha256(content):
        raise RuntimeError(
            f"{model_path} is not the file its training run logged to MLflow, so its "
            "scores can't be trusted to describe it. Train again, then promote."
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
        # Nothing to decide. Hand back the decision that put it in service, so that a
        # run interrupted after the alias moved still ends with the right record.
        logger.info("This model is already the current one")
        recorded = run.data.tags.get("promotion_decision")
        if recorded is not None:
            return json.loads(recorded)
        decision["reason"] = ALREADY_CURRENT
        return decision
    if current is not None and candidate_wape is not None:
        try:
            with tempfile.TemporaryDirectory() as tmp:
                champion = joblib.load(registry.download_model(current, tmp))
        except Exception as error:
            # Recorded as a rejection rather than raised: an operator can still put the
            # new version in service by hand with `--version`.
            logger.exception("Could not load the current model for comparison")
            matched_wape, current_wape = None, None
            obstacle: str | None = f"the current model's file could not be loaded ({error})"
        else:
            matched_wape, current_wape, obstacle = compare(candidate, champion, Path(data_dir))
        decision.update(candidate_wape=matched_wape, current_wape=current_wape)
        if obstacle is not None:
            decision["reason"] = f"no fair comparison: {obstacle}"
        else:
            decision["promoted"], decision["reason"] = decide(matched_wape, current_wape, tolerance)
    else:
        decision["promoted"], decision["reason"] = decide(candidate_wape, None, tolerance)

    # The record is written before the alias moves. If the move then fails, the next
    # run finds this model not yet current, decides again and finishes the job; the
    # other order could leave a model in service with no record of why.
    # Written synchronously whatever MLflow's logging mode, or "before" would mean
    # nothing. Promotion is assumed to run one at a time, as it does in the pipeline.
    client = MlflowClient()
    run_id = run.info.run_id
    client.set_tag(run_id, "promoted", str(decision["promoted"]).lower(), synchronous=True)
    client.set_tag(run_id, "promotion_reason", decision["reason"], synchronous=True)
    client.set_tag(run_id, "promotion_decision", json.dumps(decision), synchronous=True)
    for name in ("candidate_wape", "current_wape"):
        if decision[name] is not None:
            client.log_metric(run_id, f"promotion_{name}", decision[name], synchronous=True)
    if decision["promoted"]:
        registry.set_champion(version.version)
    return decision


def record_decision(decision: dict, path: str | Path = PROMOTION_PATH) -> bool:
    """Write the decision file. Returns whether it was written.

    The one thing not written over an existing file is the bare "already current"
    result, which only arises for a version put in service by hand and says nothing
    about how it got there.
    """
    if decision["reason"] == ALREADY_CURRENT and Path(path).exists():
        return False
    save_decision(decision, path)
    return True


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
                # Backends differ on whether a version number is text or an integer.
                "current": "*" if current and int(current.version) == int(version.version) else "",
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
    record_decision(decision)
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
