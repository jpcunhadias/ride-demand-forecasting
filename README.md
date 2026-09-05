# Ride Demand Forecasting

![Python 3.13](https://img.shields.io/badge/python-3.13-blue)
![License: MIT](https://img.shields.io/badge/license-MIT-green)
![uv](https://img.shields.io/badge/managed%20with-uv-de5fe9)

A spatiotemporal ride-demand forecasting case study for a ride-hailing
platform: EDA, zone clustering, an XGBoost baseline demand model, a
deployment strategy, and an A/B test design to evaluate it in production.

This started as a data-science take-home exercise for a ride-hailing
company, built against ride data they provided for a specific city.
Reusing a company's exact interview prompt and dataset publicly isn't
appropriate, so this is a full rebuild: the same methodology, applied
honestly end-to-end to the public Kaggle
["New York City Taxi Trip Duration"](https://www.kaggle.com/c/nyc-taxi-trip-duration/data)
dataset (1.46M trips, raw pickup/dropoff coordinates + timestamps — the
same shape as the original data). One structural difference: the original
dataset included a monetary `ride_value` field; this public one doesn't
include fares, so `ride_duration_min` stands in as the analogous per-ride
signal throughout.

## Contents

- [Key finding: the naive split silently broke the evaluation](#key-finding-the-naive-split-silently-broke-the-evaluation)
- [What the notebook covers](#what-the-notebook-covers)
- [Stack](#stack)
- [Quick start](#quick-start)
- [Project structure](#project-structure)
- [Status](#status)
- [Limitations](#limitations)

## Key finding: the naive split silently broke the evaluation

The modeling unit is `(pickup_zone, hour)`, aggregated with a straightforward
80/20 **train/test split by row count** — following the original
methodology exactly. That split looked fine until the evaluation numbers
didn't add up: raw `ride_count` (rides summed per zone-hour over the whole
split) averaged **1,214** on the training set but only **304** on the test
set — a 4x gap, for what should be the same underlying demand pattern.

The cause: ride volume is roughly uniform per day, so an 80/20 split *by
row count* isn't an 80/20 split *by time*. Train ends up covering 145
days, test only 38 — so a zone-hour's raw ride count is being compared
across two completely different-length windows. Training and evaluating
directly on that raw sum would have made the model's error look far worse
than it actually was, for a reason that has nothing to do with the model.

**Fix:** aggregate into `avg_daily_ride_count` (`ride_count / n_days` for
that split) instead of a raw sum — the actual daily rate, which is also
the quantity that matters for driver guidance in the first place ("how
many rides typically happen in this zone during this hour"). After the
fix, train and test means both land around 8 rides/day, and the model
evaluates to:

| Metric | Value |
|---|---:|
| MAE | 1.98 rides/day |
| RMSE | 3.16 rides/day |

— about a quarter of the mean and roughly 40% of the median target value,
a reasonable baseline. The full reasoning is in the notebook's
["Note: why `avg_daily_ride_count`, not raw `ride_count`"](notebooks/ride_demand_forecasting.ipynb)
cell.

## What the notebook covers

[`notebooks/ride_demand_forecasting.ipynb`](notebooks/ride_demand_forecasting.ipynb),
run end-to-end against the real dataset (not copy-pasted conclusions):

- **Data quality** — zero exact duplicates, zero non-positive durations
  (this is real production data, unlike the original's synthetic dataset,
  which had ~4,564 duplicate rows).
- **Temporal EDA** — ride volume peaks at 6 PM and troughs at 5 AM (a 6x
  gap); Friday is busiest, Monday quietest; mean trip duration peaks in
  the afternoon (~17–19 min around 2–4 PM), consistent with traffic
  congestion rather than any pricing effect.
- **Spatial cleanup** — IQR-based outlier detection flags 4–6% of rows per
  coordinate (too aggressive for geospatial data, exactly as in the
  original analysis); switching to an OSM administrative-boundary check
  (via `osmnx`) finds only **0.09%** genuine geo-outliers — GPS glitches
  landing near Sacramento, CA and out over the Atlantic — remarkably close
  to the original's 0.11%, on a totally different city and dataset.
- **Zone clustering** — KMeans (`k=40`) on pickup/dropoff coordinates.
  Pickup demand is fairly spread out (busiest single zone: 6.7% of rides;
  top 10 zones: ~51.5%), and drop-offs concentrate around Manhattan's
  commercial cores without the single dominant hub the original synthetic
  data showed.
- **Baseline model** — XGBoost regressor predicting `avg_daily_ride_count`
  per `(pickup_zone, hour)` from zone, hour, average trip duration, and
  average ride distance.
- **Deployment & driver guidance strategy** — how this would run in
  production (Airflow, Docker, FastAPI, a model registry, Grafana
  monitoring), and how drivers would see it in the app.
- **A/B test design** — population, randomization unit, metrics
  (idle time, acceptance rate, time to pickup, earnings/hour, fulfillment
  rate), and validity considerations for evaluating it against a control
  group.

## Stack

`uv`-managed. pandas, NumPy, scikit-learn, XGBoost, GeoPandas, `osmnx` +
`contextily` (OSM boundaries and basemaps), `geopy` (geodesic distance),
matplotlib/seaborn, Jupyter.

## Quick start

No Kaggle API credentials are required — this uses a manual download:

1. Download `train.csv` from Kaggle's
   [New York City Taxi Trip Duration](https://www.kaggle.com/c/nyc-taxi-trip-duration/data)
   dataset (sign-in required, free).
2. Place it at `data/raw/train.csv`.
3. ```bash
   uv sync
   uv run jupyter lab   # open notebooks/ride_demand_forecasting.ipynb,
                        # select the "ride-demand-forecasting" kernel
   ```

Re-running top to bottom takes a few minutes — most of it is a row-wise
geodesic distance calculation over ~1.4M rides and a couple of OSM/basemap
network calls.

## Project structure

```
.
├── data/
│   ├── raw/          # train.csv (Kaggle, gitignored — see Quick start)
│   └── processed/    # agg_train.csv / agg_test.csv, generated by the notebook
├── notebooks/
│   └── ride_demand_forecasting.ipynb
└── src/ride_demand_forecasting/   # scaffold placeholder for the Phase 2 service (see Status)
```

## Status

| Phase | Scope | Status |
|---|---|---|
| 1 | EDA, zone clustering, baseline XGBoost demand model, deployment strategy, A/B test design | Done |
| 2 | Productionized service — FastAPI, Docker, tests, CI | Not started |

## Limitations

- **Public data as a structural analog, not the original dataset** — same
  shape (timestamped pickup/dropoff coordinates), different city, and no
  fare data (trip duration stands in for ride value throughout — see the
  note at the top of the notebook).
- **`k=40` is a heuristic, not a validated choice** — same caveat the
  original analysis flagged: a proper elbow/silhouette pass on a
  representative sample would be the next step, not visual inspection.
- **Baseline model only** — no hyperparameter tuning, no alternative
  architectures (LightGBM, time series models) tried yet.
- **Notebook-only** — no deployed service yet; that's the explicit Phase 2
  in the Status table above.

## License

[MIT](LICENSE)
