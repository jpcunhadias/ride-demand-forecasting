# Ride Demand Forecasting

![Python 3.13](https://img.shields.io/badge/python-3.13-blue)
![License: MIT](https://img.shields.io/badge/license-MIT-green)
![uv](https://img.shields.io/badge/managed%20with-uv-de5fe9)

A ride-demand forecasting case study for a ride-hailing platform: EDA,
zone clustering, a per-zone, per-hour demand model served by a FastAPI
service and compared against naive baselines, a deployment strategy, and
an A/B test design to evaluate it in production.

The served model takes only a zone and an hour, with no date input, so it
learns a fixed demand profile per zone and hour: it predicts within 0.03
rides of the zone × hour training mean. A simple zone × hour × weekday
average beats it by 21% ([Baselines](#baselines)); adding day of week is
the next step (draft
[PR #1](https://github.com/jpcunhadias/ride-demand-forecasting/pull/1)).

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
- [Baselines](#baselines)
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
split) averaged **2,429** on the training set but only **607** on the test
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
fix, train and test means both land around 16 rides/day, and the notebook
model evaluates to:

| Metric | Value |
|---|---:|
| MAE | 3.24 rides/day |
| RMSE | 5.09 rides/day |

**These notebook numbers are leaky and are not the project's result.** The
cell scores the model with average trip duration and distance taken from
the test window itself, and with pickup zones fit on train and test rows
together. The production service removes both leaks; its honest,
day-level numbers and the baselines they are measured against are in
[Baselines](#baselines). The full reasoning for the averaging fix is in the notebook's
["Note: why `avg_daily_ride_count`, not raw `ride_count`"](notebooks/ride_demand_forecasting.ipynb)
cell. (The production service, below, sidesteps this split-window issue
entirely by training on day-level rows instead of one pre-averaged number
per zone-hour — see its own, differently-scaled metric.)

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
- **Zone clustering** — KMeans on pickup/dropoff coordinates, with `k`
  validated via elbow/silhouette analysis (silhouette peaks at **k=20**,
  half the original `k=40` visual guess). Pickup demand is fairly spread
  out (busiest single zone: 8.6% of rides; top 10 zones: ~71.5%), and
  drop-offs concentrate around Manhattan's commercial cores without the
  single dominant hub the original synthetic data showed.
- **Baseline model** — XGBoost regressor predicting `avg_daily_ride_count`
  per `(pickup_zone, hour)` from zone, hour, average trip duration, and
  average ride distance.
- **Model validation** — hyperparameter-tunes that baseline (via
  `RandomizedSearchCV` + `TimeSeriesSplit`) and compares it against a
  similarly-tuned LightGBM on a richer day-level dataset (tuned LightGBM
  wins narrowly: MAE 4.626 vs. XGBoost's 4.627), then goes further with a
  notebook-only research extension adding recency features (last
  observed count, trailing rolling mean) and two time-series baselines
  (naive rolling-mean, Holt-Winters ETS). That extension reports ~20% lower
  MAE (3.688 vs. 4.626), but it adds `day_of_week` together with the lag
  and rolling features, so the gain is confounded: a plain zone × hour ×
  weekday average with no lags already reaches MAE 3.597 on the service's
  test split (see [Baselines](#baselines)). Most of that gain is day of
  week, which needs a date input, not a stateful service.
- **Weather & holiday features** — a second notebook-only extension,
  motivated by the Winter Storm Jonas crash above. Adding daily weather
  (Open-Meteo) plus a US federal holiday flag makes the model *worse*
  under the chronological split (MAE 4.626 → 5.189), and the notebook
  diagnoses why: the test window (late May–June) has zero snow days, so
  the split can't reward learning a snow effect. A random-day split used
  purely as a diagnostic still shows a smaller negative effect, and the
  hyperparameters weren't re-tuned for the larger feature set, so weather
  is left undecided rather than rejected. The holiday flag alone gives a
  small improvement (MAE 4.626 → 4.580) that isn't confounded the same
  way, though it rests on a single holiday in the test window.
- **Deployment & driver guidance strategy** — how this would run in
  production (Airflow, Docker, FastAPI, a model registry, Grafana
  monitoring), and how drivers would see it in the app.
- **A/B test design** — population, randomization unit, metrics
  (idle time, acceptance rate, time to pickup, earnings/hour, fulfillment
  rate), and validity considerations for evaluating it against a control
  group.

## The production service

> **Served model is still the 2016 one.** Ingestion and rolling-window
> training on current NYC TLC yellow taxi data are merged on `main`, but no
> model has been trained on TLC data yet. Everything in this section, its
> metrics, the baselines and the curl outputs describe the committed
> `models/model.joblib`, trained on the first 145 days of the 2016 Kaggle
> data. Running `ride-demand-train` replaces that file — see
> [Running the service](#running-the-service).

[`src/ride_demand_forecasting/`](src/ride_demand_forecasting/) ports the
notebook's data/feature/model logic into a real package backing a FastAPI
service, with fixes and improvements that only matter once you're actually
serving predictions (the notebook itself is left untouched as the Phase-1
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
- **`k=20` zones and LightGBM hyperparameters from the notebook**: `k=20`
  is where the silhouette score peaked on the 2016 Kaggle data (it has not
  been re-checked on TLC data), and the hyperparameters come from the
  notebook's "Model Validation" search. Because the API supplies only zone
  and hour, the tuned model can only learn a fixed value per zone and hour,
  so in practice it reproduces the zone × hour training mean.
- **Day-level training rows**: rather than one pre-averaged number per
  zone-hour for the whole split (a few hundred rows total), training uses
  one row per `(pickup_zone, hour, date)` — tens of thousands of rows,
  each a real (noisier) daily observation. This also sidesteps the
  train/test window-length mismatch described in the Key Finding above:
  with no aggregation-over-a-variable-length-window step, there's nothing
  left for it to bias.

Distance is also computed with a vectorized NumPy haversine instead of the
notebook's row-wise `geopy.distance.geodesic` (city-scale difference is
<0.5%, and it drops `geopy`/`osmnx`/`geopandas`/`contextily`/`xgboost` from
the service's runtime dependencies entirely — they stay notebook-only).
With those changes, the production pipeline evaluates to:

| Metric | Value |
|---|---:|
| MAE | 4.56 rides/day |
| RMSE | 6.78 rides/day |

(Not directly comparable to the notebook's numbers above — this is a
day-level metric, measuring error against actual daily counts rather than
a pre-averaged rate, so it's a noisier, harder target by construction. It
is directly comparable to the notebook's own day-level comparison, where
the same LightGBM configuration scores MAE 4.626, RMSE 6.637. Both runs are
seeded, so the gap is not run-to-run noise; the two pipelines differ, for
example the service fits its zones on training rows only.)

A model artifact trained on the first 145 days of the 2016 data (the
chronological training split) is committed at
[`models/model.joblib`](models/model.joblib) (KMeans + `OrdinalEncoder` +
tuned `LGBMRegressor` + the zone-hour profile lookup), so the API works
right after cloning — no data download is needed to serve predictions, only
to retrain.

### Baselines

Measured on the same chronological test split as the served model: the
last 38 days of the 2016 data, one row per `(pickup_zone, hour, date)`,
18,240 rows. Each baseline uses only the training days, except seasonal
naive and persistence, which look back 7 days and 1 day.

| Predictor | MAE | RMSE |
|---|---:|---:|
| Served model (LightGBM, zone + hour) | 4.558 | 6.779 |
| Zone × hour training mean | 4.558 | 6.779 |
| **Zone × hour × weekday training mean** | **3.597** | **5.252** |
| Seasonal naive (same zone and hour, 7 days earlier) | 4.736 | 6.898 |
| Persistence (same zone and hour, 1 day earlier) | 5.324 | 8.004 |

A day-block bootstrap (2,000 resamples of test days) gives a 95% interval
of 4.25–4.91 for the model's MAE and 4.52–4.99 for seasonal naive, so
those two overlap. The served model is a learned zone × hour average: it
stays within 0.03 rides of that mean on every test row. The weekday
average, with no ML at all, is 21% better, which is the target for adding
day of week to the model.

**API:**

- `GET /health`
- `POST /predict` — body is either `{"pickup_zone": int, "hour": int}` or
  `{"lat": float, "lng": float, "hour": int}`
- `GET /rankings?hour=18&top_n=10` — zones ranked by predicted demand for
  that hour, for the driver-guidance use case described below

## Stack

`uv`-managed.

- **Service** (default install): pandas, NumPy, scikit-learn, LightGBM,
  FastAPI, Uvicorn, Pydantic, joblib.
- **Notebook only** (`uv sync --extra notebook`): XGBoost (the untuned
  baseline and the tuning comparison point), GeoPandas, `osmnx` +
  `contextily` (OSM boundaries and basemaps), `geopy` (geodesic distance),
  `statsmodels` (Holt-Winters ETS baseline), matplotlib/seaborn, Jupyter.

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

Retraining uses current NYC TLC trip data, not the Kaggle file:

```bash
uv run ride-demand-ingest  # downloads the monthly files not yet in data/raw/tlc
uv run ride-demand-train   # trains on the latest 12 months, writes models/model.joblib
```

> **This overwrites the documented model.** The metrics, baselines and
> curl outputs in this README all describe the committed 2016 model. To
> experiment without replacing it, point `RDF_MODEL_PATH` at another file.

Ingestion fetches about 60 MB per month and only what is missing, so it is
safe to re-run; `--start` and `--end` (as `YYYY-MM`) choose the range. The
TLC publishes about two months late, so the newest months are reported as
not published yet. Training evaluates with the same two-month gap between
its data and the test month, and skips evaluation when there isn't enough
history for that. It replaces the committed model, which until then is the
one trained on the 2016 data.

Configuration is via environment variables:

| Variable | Default | Purpose |
|---|---|---|
| `RDF_MODEL_PATH` | `models/model.joblib` | Artifact the API loads and the training CLI writes |
| `RDF_TLC_DATA_DIR` | `data/raw/tlc` | Where monthly trip files are downloaded to and read from |
| `RDF_MODEL_STALENESS_WARNING_DAYS` | `30` | Artifact age above which the API logs a warning at startup |

The staleness check is passive — it logs, and never blocks startup or
requests. The committed artifact was trained on 2026-09-06 against a
fixed 2016 dataset, so it will trip the default threshold; that's
expected until a model trained on current data replaces it. Raise the
variable to silence it.

## Project structure

```
.
├── data/
│   ├── raw/          # train.csv (Kaggle, for the notebook) and tlc/ (monthly
│   │                 # files, for training) — both gitignored
│   └── processed/    # agg_train.csv / agg_test.csv, generated by the notebook
├── models/
│   └── model.joblib  # committed, trained artifact backing the API
├── notebooks/
│   └── ride_demand_forecasting.ipynb
├── src/ride_demand_forecasting/
│   ├── data.py, features.py, clustering.py   # pipeline building blocks
│   ├── ingest.py                             # downloads monthly TLC trip files
│   ├── zones.py                              # TLC zone IDs to centre points
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
| 3 | Notebook research extensions — recency features, time-series baselines, weather/holiday | Done |
| 4 | Move the service to current monthly TLC data | Ingestion and rolling-window training merged; served model is still the 2016 one, not yet retrained on TLC data |
| — | Stateful serving for recency features | Deferred — needs a state store the service doesn't have yet |
| — | A `date` input with day of week, the 21% baseline gap above | In draft [PR #1](https://github.com/jpcunhadias/ride-demand-forecasting/pull/1) |
| — | Weather at serving time, an `is_holiday` input, real-time streaming | Not planned — see Limitations |

## Limitations

- **Public data as a structural analog, not the original dataset** — same
  shape (timestamped pickup/dropoff coordinates), different city, and no
  fare data (trip duration stands in for ride value throughout — see the
  note at the top of the notebook).
- **No date input leaves accuracy on the table** — the API takes only a
  zone and an hour, so the model cannot tell a Monday from a Saturday. A
  zone × hour × weekday average alone cuts MAE by 21% (4.558 → 3.597, see
  [Baselines](#baselines)). The notebook's recency extension (~20%) adds
  day of week together with lags, so it does not show that lags or a
  stateful service add much beyond day of week; that is still untested.
- **Weather and holidays stay out of the service** — the notebook's
  weather result is negative but confounded (one blizzard, all of it in
  the training window), and using weather for real would need a live
  forecast dependency. The holiday flag's small gain would need a `date`
  input the API doesn't take, and one holiday in the test window is thin
  evidence for changing the contract.
- **Batch retraining only** — no scheduled retraining/CD pipeline yet (the
  notebook's deployment strategy section sketches an Airflow-based one);
  `uv run ride-demand-train` is a manual step for now.
- **~6 months of data, one city** — the Kaggle dataset spans January-June
  2016 only, so seasonal patterns outside that window (e.g. summer, winter
  holidays) are untested; the ~5-week test window is standard for a
  chronological split but is itself drawn from that same limited span.

## License

[MIT](LICENSE)
