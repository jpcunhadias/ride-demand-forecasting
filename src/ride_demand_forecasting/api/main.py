import datetime as dt
import logging
import os
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Annotated
from zoneinfo import ZoneInfo

from fastapi import FastAPI, HTTPException, Query

from ride_demand_forecasting.api.schemas import (
    HealthResponse,
    PredictRequest,
    PredictResponse,
    RankingItem,
    RankingsResponse,
)
from ride_demand_forecasting.config import (
    MODEL_PATH,
    MODEL_STALENESS_WARNING_DAYS,
    SERVICE_TIMEZONE,
)
from ride_demand_forecasting.inference import PredictionService, UnknownZoneError

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    # Read the env var at startup time (not at import time) so tests can point
    # this at a fixture artifact via monkeypatch before the app starts up.
    model_path = os.environ.get("RDF_MODEL_PATH", str(MODEL_PATH))
    logger.info("Loading model artifact from %s", model_path)
    app.state.service = PredictionService(model_path=model_path)
    service: PredictionService = app.state.service
    age_days = service.age_days()
    logger.info(
        "Model loaded (trained_at=%s, age=%.1fd, metrics=%s)",
        service.trained_at,
        age_days,
        service.metrics,
    )
    if age_days > MODEL_STALENESS_WARNING_DAYS:
        logger.warning(
            "Model artifact is %.1f days old (warning threshold: %d days) - consider retraining",
            age_days,
            MODEL_STALENESS_WARNING_DAYS,
        )
    yield
    logger.info("Shutting down")


app = FastAPI(title="Ride Demand Forecasting", lifespan=lifespan)


def service_today() -> dt.date:
    return dt.datetime.now(ZoneInfo(SERVICE_TIMEZONE)).date()


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse(status="ok")


@app.post("/predict", response_model=PredictResponse)
def predict(request: PredictRequest) -> PredictResponse:
    service: PredictionService = app.state.service
    date = request.date or service_today()
    try:
        if request.pickup_zone is not None:
            zone = request.pickup_zone
            prediction = service.predict_zone(zone, request.hour, date)
        else:
            assert request.lat is not None and request.lng is not None
            zone, prediction = service.predict_coords(request.lat, request.lng, request.hour, date)
    except UnknownZoneError as exc:
        logger.warning("Rejected predict request: %s", exc)
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    return PredictResponse(
        pickup_zone=zone,
        hour=request.hour,
        date=date,
        predicted_avg_daily_ride_count=prediction,
    )


@app.get("/rankings", response_model=RankingsResponse)
def rankings(
    hour: Annotated[int, Query(ge=0, le=23)],
    top_n: Annotated[int | None, Query(ge=1)] = None,
    date: Annotated[dt.date | None, Query()] = None,
) -> RankingsResponse:
    service: PredictionService = app.state.service
    date = date or service_today()
    ranked = service.rank(hour, date, top_n=top_n)
    return RankingsResponse(
        hour=hour,
        date=date,
        rankings=[
            RankingItem(pickup_zone=zone, predicted_avg_daily_ride_count=pred)
            for zone, pred in ranked
        ],
    )
