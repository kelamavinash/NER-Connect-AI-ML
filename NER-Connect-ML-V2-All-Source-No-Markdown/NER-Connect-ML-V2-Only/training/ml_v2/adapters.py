"""Interfaces only: restricted or licensed datasets are never auto-downloaded."""

from datetime import datetime
from typing import Protocol, Sequence


class RoadSource(Protocol):
    source_name: str

    def road_geometries(self, bounds: tuple[float, float, float, float]) -> Sequence[object]: ...


class TerrainSource(Protocol):
    source_name: str

    def sample(self, coordinates: Sequence[tuple[float, float]]) -> Sequence[dict]: ...


class WeatherSource(Protocol):
    source_name: str

    def observations(
        self, coordinates: Sequence[tuple[float, float]], start: datetime, end: datetime
    ) -> Sequence[dict]: ...

    def forecast_vintages(
        self, coordinates: Sequence[tuple[float, float]], issued_at_or_before: datetime
    ) -> Sequence[dict]: ...


class HazardInventorySource(Protocol):
    source_name: str

    def verified_events(self, start: datetime, end: datetime) -> Sequence[dict]: ...


class IncidentSource(Protocol):
    source_name: str

    def verified_incidents(self, start: datetime, end: datetime) -> Sequence[dict]: ...


SOURCE_SCAFFOLDS = {
    "roads": "OpenStreetMap or verified government road authority",
    "landslides": "ISRO/NRSC/GSI inventories; access and license must be supplied by operator",
    "rainfall": "NASA IMERG, ERA5-Land, or Open-Meteo with archived issue times",
    "terrain": "Copernicus DEM with license/provenance recorded",
    "land_cover": "ESA WorldCover",
    "hydrology": "HydroSHEDS",
    "soil": "SoilGrids",
    "incidents": "government advisories, state disaster management, verified field reports",
}
