"""Auditable feature assembly with explicit source/time/missing provenance."""

from dataclasses import asdict, dataclass
from datetime import datetime
from typing import Any

from app.ml_v2.contract import FEATURE_BY_NAME


@dataclass(frozen=True)
class FeatureValue:
    name: str
    value: Any
    source: str
    source_timestamp: datetime | None
    units: str
    spatial_resolution: str
    version: str


class ObservationBuilder:
    def __init__(self, identifiers: dict, observation_time: datetime):
        if observation_time.tzinfo is None:
            raise ValueError("Observation time must include timezone")
        self.identifiers = dict(identifiers)
        self.observation_time = observation_time
        self.values: dict[str, FeatureValue] = {}

    def add(self, feature: FeatureValue) -> None:
        contract = FEATURE_BY_NAME.get(feature.name)
        if contract is None:
            raise ValueError(f"Unknown V2 feature: {feature.name}")
        if feature.version != contract.version or feature.units != contract.units:
            raise ValueError(f"Feature contract mismatch: {feature.name}")
        if not feature.source.strip() or not feature.spatial_resolution.strip():
            raise ValueError("Feature provenance must be explicit")
        if feature.source_timestamp and feature.source_timestamp > self.observation_time:
            raise ValueError(f"Future leakage in {feature.name}")
        if feature.name in self.values:
            raise ValueError(f"Duplicate feature value: {feature.name}")
        self.values[feature.name] = feature

    def build(self) -> tuple[dict, dict]:
        row = {**self.identifiers, "observation_time": self.observation_time}
        provenance = {}
        for name, contract in FEATURE_BY_NAME.items():
            feature = self.values.get(name)
            row[name] = None if feature is None else feature.value
            provenance[name] = (
                {
                    "available": False,
                    "missing_policy": contract.missing_policy,
                    "reason": "not supplied",
                }
                if feature is None
                else {"available": True, **asdict(feature), "source_timestamp": (
                    feature.source_timestamp.isoformat() if feature.source_timestamp else None
                )}
            )
        return row, provenance


def nearest_join(matches: list[dict]) -> dict | None:
    """Choose a precomputed projected-CRS nearest match deterministically."""
    if not matches:
        return None
    for match in matches:
        distance = match.get("distance_m")
        if not isinstance(distance, (int, float)) or distance < 0:
            raise ValueError("Spatial join requires nonnegative projected distance_m")
        if not str(match.get("crs", "")).strip():
            raise ValueError("Spatial join must record CRS")
    return min(matches, key=lambda item: (item["distance_m"], str(item.get("source_id", ""))))
