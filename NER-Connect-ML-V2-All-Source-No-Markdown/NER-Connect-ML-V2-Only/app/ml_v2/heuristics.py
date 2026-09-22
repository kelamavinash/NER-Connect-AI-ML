"""Transparent hazard-specific fallback; outputs indices, never a fake event probability."""

from app.ml_v2.contract import REQUIRED_DISRUPTION_FEATURES
from app.ml_v2.schemas import DataQuality, SegmentIntelligence, SegmentObservation


def _clamp(value: float) -> float:
    return max(0.0, min(1.0, value))


def _mean_available(values: list[float | None]) -> float | None:
    known = [value for value in values if value is not None]
    return sum(known) / len(known) if known else None


def heuristic_intelligence(segment: SegmentObservation, warnings: list[str]) -> SegmentIntelligence:
    rain = _mean_available(
        [
            None if segment.rain_6h_mm is None else segment.rain_6h_mm / 90,
            None if segment.rain_24h_mm is None else segment.rain_24h_mm / 180,
            None if segment.rain_72h_mm is None else segment.rain_72h_mm / 350,
        ]
    )
    slope = None if segment.slope_deg is None else _clamp(segment.slope_deg / 45)
    history = (
        None
        if segment.historical_landslide_density is None
        else _clamp(segment.historical_landslide_density / 5)
    )
    proximity = (
        None
        if segment.distance_to_river_m is None
        else 1 - _clamp(segment.distance_to_river_m / 2000)
    )
    low_elevation = (
        None
        if segment.elevation_m is None
        else 1 - _clamp((segment.elevation_m + 100) / 2600)
    )
    land_parts = [value for value in (rain, slope, history) if value is not None]
    flood_parts = [value for value in (rain, proximity, low_elevation) if value is not None]
    landslide = _clamp(sum(land_parts) / len(land_parts)) if land_parts else None
    flood = _clamp(sum(flood_parts) / len(flood_parts)) if flood_parts else None
    supplied = [
        value
        for name, value in segment.model_dump().items()
        if name not in {"source_timestamps"} and value is not None
    ]
    feature_fields = [
        name for name in type(segment).model_fields if name not in {"source_timestamps"}
    ]
    missing = [name for name in feature_fields if getattr(segment, name) is None]
    completeness = len(supplied) / len(feature_fields)
    required_missing = [
        name for name in REQUIRED_DISRUPTION_FEATURES if getattr(segment, name, None) is None
    ]
    if landslide is None and flood is None:
        state = "unavailable"
    elif required_missing:
        state = "partial"
    else:
        state = "live_heuristic"
    return SegmentIntelligence(
        segment_id=segment.segment_id,
        availability=state,
        method="heuristic" if state != "unavailable" else "none",
        disruption_probability=None,
        landslide_index=landslide,
        flood_index=flood,
        predicted_delay_minutes=None,
        data_quality=DataQuality(
            input_completeness=completeness,
            source_quality="unknown",
            missing_features=sorted(missing),
        ),
        model_uncertainty="unavailable",
        model_version="heuristic-v2" if state == "live_heuristic" else None,
        validated_region="not_regionally_validated",
        source_timestamps=segment.source_timestamps,
        warnings=[
            *warnings,
            "Hazard values are transparent heuristic indices, not calibrated probabilities.",
            "Delay model unavailable_data: verified delay labels are not present.",
        ],
    )
