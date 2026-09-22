"""Recorded spatial, temporal and event-group partition validation."""

from collections import defaultdict
from datetime import datetime
from math import asin, cos, radians, sin, sqrt

PARTITIONS = ("train", "calibration", "selection", "test")


def validate_split_membership(
    rows: list[dict],
    membership: dict[str, list[str]],
    *,
    enforce_unseen_district: bool = False,
) -> None:
    if set(membership) != set(PARTITIONS):
        raise ValueError("Exactly train/calibration/selection/test partitions are required")
    by_id = {row["sample_id"]: row for row in rows}
    assigned = [sample for part in PARTITIONS for sample in membership[part]]
    if len(assigned) != len(set(assigned)) or set(assigned) != set(by_id):
        raise ValueError("Split membership must assign every sample exactly once")
    grouping_keys = ["spatial_group", "storm_event_id"]
    if enforce_unseen_district:
        grouping_keys.append("district")
    for grouping_key in grouping_keys:
        owners = defaultdict(set)
        present = False
        for part in PARTITIONS:
            for sample in membership[part]:
                value = by_id[sample].get(grouping_key)
                if value:
                    present = True
                    owners[value].add(part)
        if present and any(len(parts) > 1 for parts in owners.values()):
            raise ValueError(f"{grouping_key} leaks across partitions")
    latest = {}
    earliest = {}
    for part in PARTITIONS:
        times = [by_id[sample]["observation_time"] for sample in membership[part]]
        if not times or not all(isinstance(value, datetime) for value in times):
            raise ValueError("Each partition requires parsed observation timestamps")
        earliest[part], latest[part] = min(times), max(times)
    if not (
        latest["train"] < earliest["calibration"]
        <= latest["calibration"]
        < earliest["selection"]
        <= latest["selection"]
        < earliest["test"]
    ):
        raise ValueError("Temporal partitions must progress from past to future")


def record_split_summary(
    rows: list[dict], membership: dict[str, list[str]], *, enforce_unseen_district=False
) -> dict:
    validate_split_membership(
        rows, membership, enforce_unseen_district=enforce_unseen_district
    )
    by_id = {row["sample_id"]: row for row in rows}
    return {
        part: {
            "sample_ids": list(membership[part]),
            "count": len(membership[part]),
            "temporal_start": min(by_id[s]["observation_time"] for s in membership[part]).isoformat(),
            "temporal_end": max(by_id[s]["observation_time"] for s in membership[part]).isoformat(),
            "districts": sorted({by_id[s]["district"] for s in membership[part]}),
            "storm_groups": sorted(
                {by_id[s].get("storm_event_id") for s in membership[part] if by_id[s].get("storm_event_id")}
            ),
        }
        for part in PARTITIONS
    }


def validate_spatial_buffer(
    rows: list[dict], membership: dict[str, list[str]], buffer_km: float
) -> None:
    if buffer_km <= 0:
        raise ValueError("Spatial buffer must be positive")
    by_id = {row["sample_id"]: row for row in rows}
    reference = membership["train"] + membership["calibration"] + membership["selection"]
    for test_id in membership["test"]:
        test = by_id[test_id]
        for other_id in reference:
            other = by_id[other_id]
            distance = _haversine_km(
                float(test["latitude"]),
                float(test["longitude"]),
                float(other["latitude"]),
                float(other["longitude"]),
            )
            if distance < buffer_km:
                raise ValueError("Final-test observation violates configured spatial buffer")


def _haversine_km(lat_a: float, lon_a: float, lat_b: float, lon_b: float) -> float:
    lat1, lat2 = radians(lat_a), radians(lat_b)
    d_lat, d_lon = radians(lat_b - lat_a), radians(lon_b - lon_a)
    value = sin(d_lat / 2) ** 2 + cos(lat1) * cos(lat2) * sin(d_lon / 2) ** 2
    return 2 * 6371.0088 * asin(sqrt(value))
