"""Serving schemas for per-segment V2 intelligence; Go remains policy owner."""

from datetime import datetime
from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

UnitScore = Annotated[float, Field(ge=0, le=1, allow_inf_nan=False)]
Availability = Literal["live_ml", "live_heuristic", "partial", "unavailable"]
Method = Literal["ml", "heuristic", "none"]


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, allow_inf_nan=False)


class SegmentObservation(StrictModel):
    segment_id: Annotated[str, Field(min_length=1, max_length=256)]
    observation_time: datetime
    latitude: Annotated[float, Field(ge=-90, le=90)]
    longitude: Annotated[float, Field(ge=-180, le=180)]
    state: Annotated[str, Field(min_length=1, max_length=128)]
    district: Annotated[str, Field(min_length=1, max_length=128)]
    road_id: str | None = None
    road_class: str | None = None
    slope_deg: Annotated[float | None, Field(default=None, ge=0, le=90)]
    elevation_m: Annotated[float | None, Field(default=None, ge=-500, le=9000)]
    aspect_deg: Annotated[float | None, Field(default=None, ge=0, lt=360)]
    terrain_ruggedness: Annotated[float | None, Field(default=None, ge=0)]
    curvature: float | None = None
    road_surface: str | None = None
    bridge_presence: bool | None = None
    road_width_m: Annotated[float | None, Field(default=None, gt=0)]
    land_cover: str | None = None
    soil_type: str | None = None
    geology: str | None = None
    distance_to_river_m: Annotated[float | None, Field(default=None, ge=0)]
    distance_to_drainage_m: Annotated[float | None, Field(default=None, ge=0)]
    flow_accumulation: Annotated[float | None, Field(default=None, ge=0)]
    drainage_density: Annotated[float | None, Field(default=None, ge=0)]
    floodplain_exposure: Annotated[float | None, Field(default=None, ge=0)]
    soil_permeability: Annotated[float | None, Field(default=None, ge=0)]
    historical_landslide_density: Annotated[float | None, Field(default=None, ge=0)]
    historical_flood_exposure: Annotated[float | None, Field(default=None, ge=0)]
    rain_1h_mm: Annotated[float | None, Field(default=None, ge=0)]
    rain_3h_mm: Annotated[float | None, Field(default=None, ge=0)]
    rain_6h_mm: Annotated[float | None, Field(default=None, ge=0)]
    rain_24h_mm: Annotated[float | None, Field(default=None, ge=0)]
    rain_72h_mm: Annotated[float | None, Field(default=None, ge=0)]
    forecast_rain_3h_mm: Annotated[float | None, Field(default=None, ge=0)]
    forecast_rain_6h_mm: Annotated[float | None, Field(default=None, ge=0)]
    forecast_rain_12h_mm: Annotated[float | None, Field(default=None, ge=0)]
    soil_moisture: Annotated[float | None, Field(default=None, ge=0)]
    wind_speed: Annotated[float | None, Field(default=None, ge=0)]
    wind_gust: Annotated[float | None, Field(default=None, ge=0)]
    visibility: Annotated[float | None, Field(default=None, ge=0)]
    road_condition_score: Annotated[float | None, Field(default=None, ge=0, le=100)]
    active_advisory_count: Annotated[int | None, Field(default=None, ge=0)]
    recent_incident_count: Annotated[int | None, Field(default=None, ge=0)]
    nearby_closure_count: Annotated[int | None, Field(default=None, ge=0)]
    traffic_state: str | None = None
    field_report_count: Annotated[int | None, Field(default=None, ge=0)]
    report_recency_minutes: Annotated[float | None, Field(default=None, ge=0)]
    season: str | None = None
    source_timestamps: dict[str, datetime] = Field(default_factory=dict)

    @model_validator(mode="after")
    def timestamps_are_not_future_information(self):
        future = [name for name, stamp in self.source_timestamps.items() if stamp > self.observation_time]
        if future:
            raise ValueError(f"source timestamps after observation_time: {sorted(future)}")
        return self


class IntelligenceRequest(StrictModel):
    route_id: Annotated[str, Field(min_length=1, max_length=128)]
    segments: Annotated[list[SegmentObservation], Field(min_length=1, max_length=1000)]

    @model_validator(mode="after")
    def segment_ids_are_unique(self):
        ids = [segment.segment_id for segment in self.segments]
        if len(ids) != len(set(ids)):
            raise ValueError("segment_id must be unique within a request")
        return self


class DataQuality(StrictModel):
    input_completeness: UnitScore
    source_quality: Literal["high", "medium", "low", "unknown"]
    missing_features: list[str]


class SegmentIntelligence(StrictModel):
    segment_id: str
    availability: Availability
    method: Method
    disruption_probability: UnitScore | None
    disruption_horizon_hours: Annotated[int, Field(gt=0)] = 6
    landslide_index: UnitScore | None
    flood_index: UnitScore | None
    predicted_delay_minutes: Annotated[float | None, Field(default=None, ge=0)]
    data_quality: DataQuality
    model_uncertainty: dict[str, Any] | Literal["unavailable"]
    model_version: str | None
    feature_version: Literal["2"] = "2"
    validated_region: str
    source_timestamps: dict[str, datetime]
    warnings: list[str]
    top_contributing_features: list[str] = Field(default_factory=list)


class IntelligenceResponse(StrictModel):
    route_id: str
    segments: list[SegmentIntelligence]
    warnings: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def output_segment_ids_are_unique(self):
        ids = [segment.segment_id for segment in self.segments]
        if len(ids) != len(set(ids)):
            raise ValueError("duplicate segment intelligence")
        return self
