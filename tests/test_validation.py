from pathlib import Path

import pandas as pd
import pytest

from ride_demand_forecasting.validation import backtest, summarize_backtest, validate_k


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


def test_summarize_backtest_uses_every_month_when_none_is_shared() -> None:
    results = pd.DataFrame(
        {
            "window_months": [1, 1, 2],
            "test_month": ["2026-06", "2026-07", "2026-08"],
            "mae": [1.0, 3.0, 5.0],
            "rmse": [1.0, 3.0, 5.0],
            "wape": [0.1, 0.3, 0.5],
        }
    )

    summary = summarize_backtest(results.assign(n_zones=5, smallest_zone_share=0.1)).set_index(
        "window_months"
    )

    assert summary["test_months"].tolist() == [2, 1]
    assert summary["mae"].tolist() == pytest.approx([2.0, 5.0])


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
