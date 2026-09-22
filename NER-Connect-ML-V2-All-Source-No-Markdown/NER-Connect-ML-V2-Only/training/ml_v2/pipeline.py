"""Deterministic road segmentation, weather windows, joins and label generation."""

import hashlib
import math
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Iterable, Sequence


@dataclass(frozen=True)
class RoadSegment:
    segment_id: str
    road_id: str
    start: tuple[float, float]
    end: tuple[float, float]
    sequence: int
    crs: str = "EPSG:4326"


def generate_segments(
    road_id: str, vertices: Sequence[tuple[float, float]], *, crs: str = "EPSG:4326"
) -> tuple[RoadSegment, ...]:
    if not road_id or len(vertices) < 2 or not crs:
        raise ValueError("Road ID, CRS and at least two vertices are required")
    result = []
    for sequence, (start, end) in enumerate(zip(vertices, vertices[1:])):
        if start == end or not all(math.isfinite(value) for point in (start, end) for value in point):
            raise ValueError("Road segments require distinct finite coordinates")
        identity = f"{road_id}|{sequence}|{start[0]:.7f}|{start[1]:.7f}|{end[0]:.7f}|{end[1]:.7f}|{crs}"
        segment_id = hashlib.sha256(identity.encode()).hexdigest()[:24]
        result.append(RoadSegment(segment_id, road_id, start, end, sequence, crs))
    return tuple(result)


def rainfall_windows(
    observation_time: datetime, observations: Iterable[tuple[datetime, float]]
) -> dict[str, float | None]:
    """Sum complete past-only windows; absence stays None, never zero."""
    values = list(observations)
    if observation_time.tzinfo is None:
        raise ValueError("Observation time must be timezone-aware")
    for timestamp, amount in values:
        if timestamp.tzinfo is None or timestamp > observation_time:
            raise ValueError("Weather observations must be timezone-aware and available by T")
        if not math.isfinite(amount) or amount < 0:
            raise ValueError("Invalid rainfall observation")
    result = {}
    for hours in (1, 3, 6, 24, 72):
        cutoff = observation_time - timedelta(hours=hours)
        selected = [amount for timestamp, amount in values if cutoff < timestamp <= observation_time]
        result[f"rain_{hours}h_mm"] = sum(selected) if selected else None
    return result


def forecast_window(
    observation_time: datetime,
    issued_at: datetime,
    forecast: Iterable[tuple[datetime, float]],
    hours: int,
) -> float | None:
    if issued_at > observation_time:
        raise ValueError("Forecast vintage was issued after observation time")
    end = observation_time + timedelta(hours=hours)
    values = []
    for valid_time, amount in forecast:
        if not observation_time < valid_time <= end:
            continue
        if amount < 0 or not math.isfinite(amount):
            raise ValueError("Invalid forecast rainfall")
        values.append(amount)
    return sum(values) if values else None


def disruption_label(
    segment_id: str,
    observation_time: datetime,
    incidents: Iterable[dict],
    *,
    horizon_hours: int = 6,
    documented_control: bool = False,
) -> int | None:
    end = observation_time + timedelta(hours=horizon_hours)
    matching = [
        incident
        for incident in incidents
        if incident.get("verified") is True
        and incident.get("segment_id") == segment_id
        and observation_time < incident["occurred_at"] <= end
        and incident.get("logistics_relevant") is True
    ]
    if matching:
        return 1
    return 0 if documented_control else None
