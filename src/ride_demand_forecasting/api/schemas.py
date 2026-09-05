from pydantic import BaseModel, Field, model_validator

from ride_demand_forecasting.config import NYC_LAT_MAX, NYC_LAT_MIN, NYC_LNG_MAX, NYC_LNG_MIN


class PredictRequest(BaseModel):
    hour: int = Field(ge=0, le=23, description="Hour of day, 0-23")
    pickup_zone: int | None = Field(default=None, description="Internal pickup zone id")
    lat: float | None = Field(default=None, description="Pickup latitude")
    lng: float | None = Field(default=None, description="Pickup longitude")

    @model_validator(mode="after")
    def check_exactly_one_location(self) -> "PredictRequest":
        has_zone = self.pickup_zone is not None
        has_coords = self.lat is not None and self.lng is not None
        if has_zone == has_coords:
            raise ValueError(
                "Provide exactly one of `pickup_zone` or both `lat`/`lng`, not neither or both."
            )
        if (self.lat is None) != (self.lng is None):
            raise ValueError("`lat` and `lng` must be provided together.")
        if has_coords:
            assert self.lat is not None and self.lng is not None
            if not (NYC_LAT_MIN <= self.lat <= NYC_LAT_MAX):
                raise ValueError(f"`lat` must be within [{NYC_LAT_MIN}, {NYC_LAT_MAX}]")
            if not (NYC_LNG_MIN <= self.lng <= NYC_LNG_MAX):
                raise ValueError(f"`lng` must be within [{NYC_LNG_MIN}, {NYC_LNG_MAX}]")
        return self


class PredictResponse(BaseModel):
    pickup_zone: int
    hour: int
    predicted_avg_daily_ride_count: float


class RankingItem(BaseModel):
    pickup_zone: int
    predicted_avg_daily_ride_count: float


class RankingsResponse(BaseModel):
    hour: int
    rankings: list[RankingItem]


class HealthResponse(BaseModel):
    status: str
