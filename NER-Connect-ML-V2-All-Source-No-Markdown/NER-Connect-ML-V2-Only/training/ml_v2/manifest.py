"""Dataset manifest V2 validation and content fingerprinting."""

import hashlib
import json
from datetime import datetime
from pathlib import Path

from app.ml_v2.contract import FEATURE_BY_NAME

SCHEMA_VERSION = "2"
REQUIRED_FIELDS = (
    "dataset_name",
    "schema_version",
    "created_at",
    "kind",
    "geographic_coverage",
    "temporal_coverage",
    "sources",
    "licenses",
    "feature_definitions",
    "positive_target_definition",
    "negative_control_sampling_method",
    "disruption_definition",
    "label_source",
    "deduplication_rules",
    "spatial_resolution",
    "road_segment_generation_method",
    "weather_alignment_method",
    "missing_data_policy",
    "dataset_fingerprint_sha256",
    "split_membership",
    "known_limitations",
)


def sha256_file(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def validate_manifest(manifest: dict, *, dataset_path: Path | None = None) -> dict:
    missing = [field for field in REQUIRED_FIELDS if field not in manifest]
    if missing:
        raise ValueError(f"Manifest V2 missing fields: {missing}")
    if manifest["schema_version"] != SCHEMA_VERSION:
        raise ValueError("Dataset manifest schema_version must be 2")
    if manifest["kind"] not in {"real_ner_labels", "research_external", "test_fixture"}:
        raise ValueError("Dataset kind must describe evidence, not a filename")
    created_at = datetime.fromisoformat(str(manifest["created_at"]).replace("Z", "+00:00"))
    if created_at.tzinfo is None:
        raise ValueError("Manifest created_at must include timezone")
    if not manifest["sources"] or not manifest["licenses"]:
        raise ValueError("Dataset sources and licenses must be explicit")
    if not str(manifest["negative_control_sampling_method"]).strip():
        raise ValueError("Undocumented observations cannot be treated as negative controls")
    fingerprint = manifest["dataset_fingerprint_sha256"]
    if not isinstance(fingerprint, str) or len(fingerprint) != 64:
        raise ValueError("Dataset fingerprint must be SHA-256")
    int(fingerprint, 16)
    if dataset_path is not None and sha256_file(dataset_path) != fingerprint:
        raise ValueError("Dataset fingerprint mismatch")
    for name, definition in manifest["feature_definitions"].items():
        if name not in FEATURE_BY_NAME:
            raise ValueError(f"Unknown V2 feature: {name}")
        for key in (
            "source",
            "units",
            "temporal_status",
            "missing_policy",
            "transformation",
            "spatial_resolution",
            "version",
        ):
            if not str(definition.get(key, "")).strip():
                raise ValueError(f"Incomplete feature definition: {name}.{key}")
    splits = manifest["split_membership"]
    if set(splits) != {"train", "calibration", "selection", "test"}:
        raise ValueError("Split membership must include four isolated partitions")
    memberships = [sample for values in splits.values() for sample in values]
    if any(not values for values in splits.values()):
        raise ValueError("Each split partition must be nonempty")
    if len(memberships) != len(set(memberships)):
        raise ValueError("A sample appears in more than one partition")
    return manifest


def load_manifest(path: Path, *, dataset_path: Path | None = None) -> dict:
    return validate_manifest(json.loads(path.read_text(encoding="utf-8")), dataset_path=dataset_path)


def write_manifest(path: Path, manifest: dict) -> None:
    validate_manifest(manifest)
    with path.open("x", encoding="utf-8") as stream:
        json.dump(manifest, stream, indent=2, allow_nan=False, sort_keys=True)
