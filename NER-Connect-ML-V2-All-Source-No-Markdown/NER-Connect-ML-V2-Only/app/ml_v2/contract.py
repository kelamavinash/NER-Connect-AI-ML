"""Versioned, auditable feature definitions for road-segment observations."""

from dataclasses import asdict, dataclass
from typing import Literal

FEATURE_VERSION = "2"
TARGET_HORIZON_HOURS = 6


@dataclass(frozen=True)
class FeatureDefinition:
    name: str
    source: str
    units: str
    temporal_status: str
    missing_policy: Literal["reject_ml", "flag_missing", "not_available"]
    transformation: str
    spatial_resolution: str
    version: str = FEATURE_VERSION
    required_for_disruption: bool = False


FEATURES = (
    FeatureDefinition("slope_deg", "DEM derivative", "degrees", "static", "reject_ml", "none", "source dependent", required_for_disruption=True),
    FeatureDefinition("elevation_m", "DEM", "metres", "static", "reject_ml", "none", "source dependent", required_for_disruption=True),
    FeatureDefinition("aspect_deg", "DEM derivative", "degrees clockwise from north", "static", "flag_missing", "cyclic encoding in model pipeline", "source dependent"),
    FeatureDefinition("terrain_ruggedness", "DEM derivative", "index", "static", "flag_missing", "documented source algorithm", "source dependent"),
    FeatureDefinition("curvature", "DEM derivative", "1/metre", "static", "flag_missing", "none", "source dependent"),
    FeatureDefinition("road_class", "OpenStreetMap or road authority", "category", "static/versioned", "reject_ml", "native categorical", "road segment", required_for_disruption=True),
    FeatureDefinition("road_surface", "OpenStreetMap or road authority", "category", "static/versioned", "flag_missing", "native categorical", "road segment"),
    FeatureDefinition("bridge_presence", "OpenStreetMap or road authority", "boolean", "static/versioned", "flag_missing", "none", "road segment"),
    FeatureDefinition("road_width_m", "road authority or field survey", "metres", "static/versioned", "flag_missing", "none", "road segment"),
    FeatureDefinition("land_cover", "ESA WorldCover", "category", "static/versioned", "flag_missing", "native categorical", "source dependent"),
    FeatureDefinition("soil_type", "SoilGrids or verified regional source", "category", "static/versioned", "flag_missing", "native categorical", "source dependent"),
    FeatureDefinition("geology", "GSI or verified regional source", "category", "static/versioned", "flag_missing", "native categorical", "source dependent"),
    FeatureDefinition("distance_to_river_m", "HydroSHEDS or verified hydrography", "metres", "static/versioned", "flag_missing", "nearest-distance in projected CRS", "source dependent"),
    FeatureDefinition("distance_to_drainage_m", "HydroSHEDS or verified drainage", "metres", "static/versioned", "flag_missing", "nearest-distance in projected CRS", "source dependent"),
    FeatureDefinition("flow_accumulation", "HydroSHEDS or derived DEM hydrology", "source-defined index", "static/versioned", "flag_missing", "documented hydrology algorithm", "source dependent"),
    FeatureDefinition("drainage_density", "HydroSHEDS or verified drainage", "km/km2", "static/versioned", "flag_missing", "documented analysis window", "documented buffer"),
    FeatureDefinition("floodplain_exposure", "verified floodplain map", "index", "static/versioned", "flag_missing", "documented spatial overlay", "source dependent"),
    FeatureDefinition("soil_permeability", "SoilGrids or verified regional source", "source-defined", "static/versioned", "flag_missing", "documented source mapping", "source dependent"),
    FeatureDefinition("historical_landslide_density", "verified pre-observation inventory", "events/km2", "as-of observation_time", "flag_missing", "past-only spatial density", "documented buffer"),
    FeatureDefinition("historical_flood_exposure", "verified pre-observation inventory", "index", "as-of observation_time", "flag_missing", "past-only spatial exposure", "documented source"),
    FeatureDefinition("rain_1h_mm", "timestamped weather observations", "mm", "lookback ending at observation_time", "flag_missing", "sum of available observations", "weather grid"),
    FeatureDefinition("rain_3h_mm", "timestamped weather observations", "mm", "lookback ending at observation_time", "flag_missing", "sum of available observations", "weather grid"),
    FeatureDefinition("rain_6h_mm", "timestamped weather observations", "mm", "lookback ending at observation_time", "reject_ml", "sum of available observations", "weather grid", required_for_disruption=True),
    FeatureDefinition("rain_24h_mm", "timestamped weather observations", "mm", "lookback ending at observation_time", "reject_ml", "sum of available observations", "weather grid", required_for_disruption=True),
    FeatureDefinition("rain_72h_mm", "timestamped weather observations", "mm", "lookback ending at observation_time", "flag_missing", "sum of available observations", "weather grid"),
    FeatureDefinition("forecast_rain_3h_mm", "forecast issued at/before observation_time", "mm", "forecast horizon from observation_time", "flag_missing", "sum from preserved forecast vintage", "forecast grid"),
    FeatureDefinition("forecast_rain_6h_mm", "forecast issued at/before observation_time", "mm", "forecast horizon from observation_time", "flag_missing", "sum from preserved forecast vintage", "forecast grid"),
    FeatureDefinition("forecast_rain_12h_mm", "forecast issued at/before observation_time", "mm", "forecast horizon from observation_time", "flag_missing", "sum from preserved forecast vintage", "forecast grid"),
    FeatureDefinition("soil_moisture", "timestamped provider", "provider-defined", "available at/before observation_time", "flag_missing", "none", "source dependent"),
    FeatureDefinition("wind_speed", "timestamped weather provider", "km/h", "available at/before observation_time", "flag_missing", "none", "weather grid"),
    FeatureDefinition("wind_gust", "timestamped weather provider", "km/h", "available at/before observation_time", "flag_missing", "none", "weather grid"),
    FeatureDefinition("visibility", "timestamped weather provider", "metres", "available at/before observation_time", "flag_missing", "none", "weather grid"),
    FeatureDefinition("road_condition_score", "road authority or verified field report", "0-100; higher is better", "as-of observation_time", "reject_ml", "none", "road segment", required_for_disruption=True),
    FeatureDefinition("active_advisory_count", "government/verified advisories", "count", "active at observation_time", "flag_missing", "spatiotemporal join", "documented buffer"),
    FeatureDefinition("recent_incident_count", "verified incident feed", "count", "past-only window", "flag_missing", "spatiotemporal count", "documented buffer"),
    FeatureDefinition("nearby_closure_count", "verified closure feed", "count", "active at observation_time", "flag_missing", "spatiotemporal count", "documented buffer"),
    FeatureDefinition("traffic_state", "timestamped traffic provider", "category", "available at/before observation_time", "flag_missing", "native categorical", "road segment"),
    FeatureDefinition("field_report_count", "verified field reports", "count", "past-only window", "flag_missing", "spatiotemporal count", "documented buffer"),
    FeatureDefinition("report_recency_minutes", "verified field reports", "minutes", "as-of observation_time", "flag_missing", "observation minus latest report", "road segment"),
    FeatureDefinition("season", "derived from observation_time", "category", "at observation_time", "flag_missing", "documented regional season mapping", "road segment observation"),
)

FEATURE_BY_NAME = {feature.name: feature for feature in FEATURES}
DISRUPTION_FEATURE_ORDER = tuple(feature.name for feature in FEATURES)
REQUIRED_DISRUPTION_FEATURES = tuple(
    feature.name for feature in FEATURES if feature.required_for_disruption
)
CATEGORICAL_FEATURES = (
    "state",
    "district",
    "road_class",
    "road_surface",
    "geology",
    "soil_type",
    "land_cover",
    "traffic_state",
    "season",
)

TARGET_FIELDS = ("disruption_next_6h", "delay_minutes")


def contract_as_dict() -> dict:
    return {
        "feature_version": FEATURE_VERSION,
        "grain": "road segment x observation time",
        "target_horizon_hours": TARGET_HORIZON_HOURS,
        "features": [asdict(feature) for feature in FEATURES],
        "categorical_features": list(CATEGORICAL_FEATURES),
        "target_fields": list(TARGET_FIELDS),
    }
