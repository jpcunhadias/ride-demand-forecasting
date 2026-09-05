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
- [The production service](#the-production-service)
- [Stack](#stack)
- [Quick start](#quick-start)
- [Running the service](#running-the-service)
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
| MAE | 1.72 rides/day |
| RMSE | 2.67 rides/day |

— about a fifth of the mean and roughly 30% of the median target value,
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
  congestion rather than any pricing effect. The daily series has one
  sharp crash — January 23, 2016, down to ~1,600 rides from a normal
  7,000–9,500 — that lines up exactly with Winter Storm Jonas ("Snowzilla"),
  a real external demand shock the current feature set can't see coming.
- **Spatial cleanup** — IQR-based outlier detection flags 4–6% of rows per
  coordinate (too aggressive for geospatial data, exactly as in the
  original analysis); switching to an OSM administrative-boundary check
  (via `osmnx`) finds only **0.09%** genuine geo-outliers — GPS glitches
  landing near Sacramento, CA and out over the Atlantic — remarkably close
  to the original's 0.11%, on a totally different city and dataset. Unlike
  the original (where only drop-off coordinates had glitches), this
  dataset has them in pickup coordinates too, so the cleaning filter checks
  all four coordinates rather than just drop-off.
- **Zone clustering** — KMeans (`k=40`) on pickup/dropoff coordinates.
  Pickup demand is fairly spread out (busiest single zone: 6.1% of rides;
  top 10 zones: ~50.4%), and drop-offs concentrate around Manhattan's
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

## The production service

[`src/ride_demand_forecasting/`](src/ride_demand_forecasting/) ports the
notebook's data/feature/model logic into a real package backing a FastAPI
service, with two fixes that only matter once you're actually serving
predictions (the notebook itself is left untouched as the Phase-1
exploratory record):

- **Spatial leakage**: the notebook fits its pickup-zone KMeans on
  train+test coordinates combined. [`train.py`](src/ride_demand_forecasting/train.py)
  fits it on **train-only** coordinates, then assigns zones to the test set
  with the already-fitted model.
- **Circular features**: the notebook trains on `avg_ride_duration_min` /
  `avg_ride_distance` as if they were known ahead of time — but those are
  outcomes of rides that haven't happened yet, so a real caller can't
  supply them for a future prediction window. The training pipeline instead
  builds a `(pickup_zone, hour) -> historical avg duration/distance` lookup
  from training data, and the service uses that internally — the API only
  needs `pickup_zone` (or raw `lat`/`lng`, mapped to a zone via the
  persisted KMeans model) and `hour`.

Distance is also computed with a vectorized NumPy haversine instead of the
notebook's row-wise `geopy.distance.geodesic` (city-scale difference is
<0.5%, and it drops `geopy`/`osmnx`/`geopandas`/`contextily` from the
service's runtime dependencies entirely — they stay notebook-only). With
those fixes, the production pipeline evaluates to:

| Metric | Value |
|---|---:|
| MAE | 2.10 rides/day |
| RMSE | 3.24 rides/day |

(Higher than the notebook's 1.72/2.67 — expected, since the leakage fix
removes an unrealistic advantage the notebook's baseline had.)

A model artifact trained on the full dataset is committed at
[`models/model.joblib`](models/model.joblib) (KMeans + `OrdinalEncoder` +
`XGBRegressor` + the zone-hour profile lookup), so the API works right
after cloning — no Kaggle download needed to serve predictions, only to
retrain.

**API:**

- `GET /health`
- `POST /predict` — body is either `{"pickup_zone": int, "hour": int}` or
  `{"lat": float, "lng": float, "hour": int}`
- `GET /rankings?hour=18&top_n=10` — zones ranked by predicted demand for
  that hour, for the driver-guidance use case described below

## Stack

`uv`-managed.

- **Service** (default install): pandas, NumPy, scikit-learn, XGBoost,
  FastAPI, Uvicorn, Pydantic, joblib.
- **Notebook only** (`uv sync --extra notebook`): GeoPandas, `osmnx` +
  `contextily` (OSM boundaries and basemaps), `geopy` (geodesic distance),
  matplotlib/seaborn, Jupyter.

## Quick start

No Kaggle API credentials are required — this uses a manual download:

1. Download `train.csv` from Kaggle's
   [New York City Taxi Trip Duration](https://www.kaggle.com/c/nyc-taxi-trip-duration/data)
   dataset (sign-in required, free).
2. Place it at `data/raw/train.csv`.
3. ```bash
   uv sync --extra notebook
   uv run jupyter lab   # open notebooks/ride_demand_forecasting.ipynb,
                        # select the "ride-demand-forecasting" kernel
   ```

Re-running top to bottom takes a few minutes — most of it is a row-wise
geodesic distance calculation over ~1.4M rides and a couple of OSM/basemap
network calls.

## Running the service

A trained artifact is already committed, so this works without the Kaggle
dataset:

```bash
uv sync
uv run uvicorn ride_demand_forecasting.api.main:app --reload
```

```bash
curl -X POST localhost:8000/predict -H 'Content-Type: application/json' \
  -d '{"pickup_zone": 5, "hour": 18}'

curl -X POST localhost:8000/predict -H 'Content-Type: application/json' \
  -d '{"lat": 40.75, "lng": -73.98, "hour": 18}'

curl 'localhost:8000/rankings?hour=18&top_n=10'
```

Or with Docker:

```bash
docker build -t ride-demand-forecasting .
docker run -p 8000:8000 ride-demand-forecasting
```

To retrain against your own copy of `data/raw/train.csv`:

```bash
uv run ride-demand-train   # writes models/model.joblib
```

## Project structure

```
.
├── data/
│   ├── raw/          # train.csv (Kaggle, gitignored — see Quick start)
│   └── processed/    # agg_train.csv / agg_test.csv, generated by the notebook
├── models/
│   └── model.joblib  # committed, trained artifact backing the API
├── notebooks/
│   └── ride_demand_forecasting.ipynb
├── src/ride_demand_forecasting/
│   ├── data.py, features.py, clustering.py   # pipeline building blocks
│   ├── train.py                              # training entry point
│   ├── inference.py                          # loads the artifact, serves predictions
│   └── api/                                  # FastAPI app
├── tests/
├── Dockerfile
└── .github/workflows/ci.yml
```

## Status

| Phase | Scope | Status |
|---|---|---|
| 1 | EDA, zone clustering, baseline XGBoost demand model, deployment strategy, A/B test design | Done |
| 2 | Productionized service — FastAPI, Docker, tests, CI | Done |

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
- **Batch retraining only** — no scheduled retraining/CD pipeline yet (the
  notebook's deployment strategy section sketches an Airflow-based one);
  `uv run ride-demand-train` is a manual step for now.

## License

[MIT](LICENSE)
