from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from ride_demand_forecasting import validation
from ride_demand_forecasting.train import load_months
from ride_demand_forecasting.validation import (
    backtest,
    sample_lgbm_params,
    summarize_backtest,
    tune,
    validate_k,
)


@pytest.fixture(scope="module")
def backtest_results(tlc_data_dir: Path) -> pd.DataFrame:
    # Five months on disk (2026-04..08) and a two-month gap: 2026-07 can be tested
    # with windows ending 2026-05, and 2026-08 with windows ending 2026-06.
    return backtest(data_dir=tlc_data_dir, windows=(1, 2, 3), zone_counts=(5,), n_test_months=2)


def test_backtest_runs_every_window_that_is_fully_on_disk(backtest_results: pd.DataFrame) -> None:
    runs = set(zip(backtest_results["window_months"], backtest_results["test_month"], strict=True))

    # A three-month window before 2026-07 would need 2026-03, which isn't there.
    assert runs == {
        (1, "2026-07"),
        (2, "2026-07"),
        (1, "2026-08"),
        (2, "2026-08"),
        (3, "2026-08"),
    }
    assert (backtest_results[["mae", "rmse", "wape"]] >= 0).all().all()
    assert (backtest_results["n_test_rows"] > 0).all()


def test_summarize_backtest_compares_windows_on_the_same_test_months(
    backtest_results: pd.DataFrame,
) -> None:
    summary = summarize_backtest(backtest_results).set_index("window_months")

    # Only 2026-08 was run for all three windows.
    assert summary["test_months"].tolist() == [1, 1, 1]
    august = backtest_results[backtest_results["test_month"] == "2026-08"].set_index(
        "window_months"
    )
    assert summary["mae"].tolist() == pytest.approx(august["mae"].tolist())


def test_summarize_backtest_refuses_to_compare_settings_with_no_shared_month() -> None:
    # Each window has one test month, but not the same one.
    results = pd.DataFrame(
        {
            "window_months": [1, 2],
            "n_zones": [5, 5],
            "test_month": ["2026-07", "2026-08"],
            "mae": [1.0, 5.0],
            "rmse": [1.0, 5.0],
            "wape": [0.1, 0.5],
            "smallest_zone_share": [0.1, 0.1],
        }
    )

    with pytest.raises(ValueError, match="No test month was run for every setting"):
        summarize_backtest(results)


def test_summarize_backtest_is_not_fooled_by_repeated_rows() -> None:
    # Two rows per month, but each month has only one of the two settings.
    results = pd.DataFrame(
        {
            "window_months": [1, 1, 2, 2],
            "n_zones": [5, 5, 5, 5],
            "test_month": ["2026-07", "2026-07", "2026-08", "2026-08"],
            "mae": 1.0,
            "rmse": 1.0,
            "wape": 0.1,
            "smallest_zone_share": 0.1,
        }
    )

    with pytest.raises(ValueError, match="No test month was run for every setting"):
        summarize_backtest(results)


def test_backtest_runs_a_repeated_setting_once(tlc_data_dir: Path) -> None:
    results = backtest(data_dir=tlc_data_dir, windows=(1, 1), zone_counts=(5, 5), n_test_months=1)

    assert results[["window_months", "n_zones", "test_month"]].values.tolist() == [
        [1, 5, "2026-08"]
    ]


def test_validation_commands_fail_clearly_when_nothing_is_ingested(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="ride-demand-ingest"):
        backtest(data_dir=tmp_path)
    with pytest.raises(FileNotFoundError, match="ride-demand-ingest"):
        validate_k(data_dir=tmp_path)


def test_backtest_compares_zone_counts_on_the_same_windows(tlc_data_dir: Path) -> None:
    # 30 synthetic taxi zones, so 40 pickup zones can't be formed and is skipped.
    results = backtest(
        data_dir=tlc_data_dir, windows=(2,), zone_counts=(3, 10, 40), n_test_months=1
    )

    assert results["n_zones"].tolist() == [3, 10]
    assert results["test_month"].unique().tolist() == ["2026-08"]
    assert results["wape"].between(0, 5).all()
    # More zones means each holds less of the demand.
    shares = results.set_index("n_zones")["smallest_zone_share"]
    assert 0 < shares[10] < shares[3] < 1

    summary = summarize_backtest(results)
    assert summary[["window_months", "n_zones"]].values.tolist() == [[2, 3], [2, 10]]
    assert summary["wape"].tolist() == pytest.approx(results["wape"].tolist())


def test_validate_k_scores_each_candidate(tlc_data_dir: Path) -> None:
    # 30 synthetic zones: k=30 and above leave nothing to cluster, so they're skipped.
    results = validate_k(data_dir=tlc_data_dir, k_values=[2, 5, 10, 30, 40])

    assert results["k"].tolist() == [2, 5, 10]
    assert results["inertia"].is_monotonic_decreasing
    assert results["silhouette"].between(-1, 1).all()
    assert (results["smallest_zone_share"] > 0).all()
    assert (results["smallest_zone_share"] <= results["largest_zone_share"]).all()


def test_validate_k_uses_the_same_calendar_window_as_training(
    tmp_path: Path, make_tlc_data_dir, monkeypatch: pytest.MonkeyPatch
) -> None:
    data_dir = make_tlc_data_dir(tmp_path / "tlc", ["2026-05", "2026-08"])
    loaded_months: list[str] = []

    def spy(months, *args, **kwargs):
        loaded_months.extend(str(month) for month in months)
        return load_months(months, *args, **kwargs)

    monkeypatch.setattr(validation, "load_months", spy)

    validate_k(data_dir=data_dir, k_values=[3], window_months=2)

    # A two-month window ending in August is July and August. May is on disk but
    # outside it, and must not stand in for the missing July.
    assert loaded_months == ["2026-08"]


def small_lgbm_params(rng: np.random.Generator) -> dict:
    """A stand-in for the real search space with small, quick-to-fit models."""
    return {
        "n_estimators": int(rng.integers(10, 40)),
        "num_leaves": 15,
        "learning_rate": 0.1,
        "subsample": 0.8,
        "subsample_freq": 1,
        "colsample_bytree": 0.8,
    }


def quick_tune(data_dir: Path, n_trials: int) -> pd.DataFrame:
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(validation, "sample_lgbm_params", small_lgbm_params)
        return tune(
            data_dir=data_dir, n_trials=n_trials, n_test_months=2, window_months=2, n_zones=5
        )


@pytest.fixture(scope="module")
def tuning_results(tlc_data_dir: Path) -> pd.DataFrame:
    return quick_tune(tlc_data_dir, n_trials=3)


def test_tune_scores_the_current_configuration_and_every_trial(
    tuning_results: pd.DataFrame,
) -> None:
    assert sorted(tuning_results["trial"]) == ["current", "trial-1", "trial-2", "trial-3"]
    assert tuning_results["wape"].is_monotonic_increasing
    assert (tuning_results["folds"] == 2).all()
    sampled = tuning_results[tuning_results["trial"] != "current"]
    assert all(params["subsample_freq"] == 1 for params in sampled["params"])


def test_tune_scores_the_current_configuration_like_the_backtest(
    tlc_data_dir: Path, tuning_results: pd.DataFrame
) -> None:
    # Same folds, same zones, same hyperparameters: the two commands must agree.
    backtested = summarize_backtest(
        backtest(data_dir=tlc_data_dir, windows=(2,), zone_counts=(5,), n_test_months=2)
    ).iloc[0]
    current = tuning_results[tuning_results["trial"] == "current"].iloc[0]

    assert current["wape"] == pytest.approx(backtested["wape"])
    assert current["mae"] == pytest.approx(backtested["mae"])


def test_tune_is_repeatable(tlc_data_dir: Path, tuning_results: pd.DataFrame) -> None:
    again = quick_tune(tlc_data_dir, n_trials=1)

    # The same seed draws the same first configuration and scores it the same.
    first = tuning_results[tuning_results["trial"] == "trial-1"].iloc[0]
    repeated = again[again["trial"] == "trial-1"].iloc[0]
    assert repeated["params"] == first["params"]
    assert repeated["wape"] == pytest.approx(first["wape"])


def test_tune_fails_clearly_without_enough_history(tmp_path: Path, make_tlc_data_dir) -> None:
    data_dir = make_tlc_data_dir(tmp_path / "tlc", ["2026-08"])

    with pytest.raises(ValueError, match="nothing can be tuned"):
        tune(data_dir=data_dir, n_trials=1, n_zones=5)


def test_sample_lgbm_params_stays_inside_the_search_space() -> None:
    rng = np.random.default_rng(0)

    for _ in range(200):
        params = sample_lgbm_params(rng)
        assert 100 <= params["n_estimators"] <= 600
        assert 15 <= params["num_leaves"] <= 255
        assert 0.01 <= params["learning_rate"] <= 0.2
        assert 0.6 <= params["subsample"] <= 1.0
        assert 0.6 <= params["colsample_bytree"] <= 1.0
