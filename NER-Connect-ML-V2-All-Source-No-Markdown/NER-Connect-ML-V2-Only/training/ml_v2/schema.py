"""Road segment x observation-time dataset parsing and scientific guards."""

import csv
import math
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from app.ml_v2.contract import DISRUPTION_FEATURE_ORDER, FEATURE_BY_NAME
from training.ml_v2.manifest import load_manifest

IDENTIFIERS = (
    "sample_id",
    "segment_id",
    "observation_time",
    "latitude",
    "longitude",
    "state",
    "district",
    "road_id",
    "road_class",
)


@dataclass(frozen=True)
class TrainingTable:
    rows: tuple[dict, ...]
    manifest: dict


def parse_timestamp(value: str) -> datetime:
    result = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if result.tzinfo is None:
        raise ValueError("Timestamps must include a timezone")
    return result.astimezone(timezone.utc)


def _optional_number(row: dict, name: str) -> float | None:
    raw = row.get(name, "")
    if raw in (None, ""):
        return None
    value = float(raw)
    if not math.isfinite(value):
        raise ValueError(f"Nonfinite value: {name}")
    return value


def validate_row(row: dict, horizon_hours: int = 6) -> dict:
    for name in IDENTIFIERS:
        if not str(row.get(name, "")).strip():
            raise ValueError(f"Missing identifier: {name}")
    observation = parse_timestamp(row["observation_time"])
    for name, value in row.items():
        if name.endswith("_source_time") or name.endswith("_issued_at"):
            if value and parse_timestamp(value) > observation:
                raise ValueError(f"Future leakage in {name}")
    if any(name in row for name in ("event_label", "future_disruption", "label_source_text")):
        raise ValueError("Event-label fields may not enter predictor rows")
    label_name = f"disruption_next_{horizon_hours}h"
    label = row.get(label_name, "")
    if label not in ("", None, "0", "1", 0, 1):
        raise ValueError("Disruption label must be explicit binary or missing")
    if label in ("0", 0) and str(row.get("control_sampling_status", "")) != "documented_control":
        raise ValueError("Negative rows require documented control sampling")
    delay = _optional_number(row, "delay_minutes")
    if delay is not None and delay < 0:
        raise ValueError("Delay cannot be negative")
    parsed = dict(row)
    parsed["observation_time"] = observation
    parsed[label_name] = None if label in ("", None) else int(label)
    parsed["delay_minutes"] = delay
    for feature in DISRUPTION_FEATURE_ORDER:
        if feature in row and feature not in {
            "road_class",
            "road_surface",
            "land_cover",
            "soil_type",
            "geology",
            "traffic_state",
            "season",
        }:
            parsed[feature] = _optional_number(row, feature)
    return parsed


def load_training_table(csv_path: Path, manifest_path: Path) -> TrainingTable:
    manifest = load_manifest(manifest_path, dataset_path=csv_path)
    rows = []
    with csv_path.open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        fields = set(reader.fieldnames or ())
        required = set(IDENTIFIERS) | {"disruption_next_6h", "control_sampling_status"}
        if not required.issubset(fields):
            raise ValueError(f"Training table missing columns: {sorted(required - fields)}")
        undocumented = [
            name
            for name in FEATURE_BY_NAME
            if name in fields and name not in manifest["feature_definitions"]
        ]
        if undocumented:
            raise ValueError(f"Dataset features lack manifest definitions: {undocumented}")
        for line, row in enumerate(reader, 2):
            try:
                rows.append(validate_row(row))
            except (TypeError, ValueError) as exc:
                raise ValueError(f"Invalid training row {line}: {exc}") from exc
    ids = [row["sample_id"] for row in rows]
    grain = [(row["segment_id"], row["observation_time"]) for row in rows]
    if len(ids) != len(set(ids)) or len(grain) != len(set(grain)):
        raise ValueError("Duplicate sample or segment/observation grain")
    if not rows:
        raise ValueError("Training table is empty")
    return TrainingTable(tuple(rows), manifest)
