"""Artifact registry V2 metadata; candidates are never approved automatically."""

import hashlib
import importlib.metadata
import json
import math
import platform
from datetime import datetime, timezone
from pathlib import Path

import joblib

from app.ml_v2.contract import FEATURE_VERSION

REQUIRED = (
    "model_name",
    "model_version",
    "task",
    "feature_version",
    "training_timestamp",
    "dataset_fingerprint",
    "dataset_provenance",
    "training_geography",
    "validation_geography",
    "temporal_coverage",
    "target_horizon_hours",
    "selected_threshold",
    "calibration_method",
    "metrics",
    "library_versions",
    "artifact_sha256",
    "serving_approved",
    "review_notes",
    "validated_region",
    "input_support",
)


def validate_metadata(metadata: dict, *, require_approved: bool = False) -> None:
    missing = [name for name in REQUIRED if name not in metadata]
    if missing:
        raise ValueError(f"Artifact V2 metadata missing: {missing}")
    if metadata["feature_version"] != FEATURE_VERSION:
        raise ValueError("Artifact feature contract is incompatible")
    if metadata["task"] not in {"road_segment_disruption", "delay", "landslide", "flood"}:
        raise ValueError("Unknown artifact task")
    trained_at = datetime.fromisoformat(str(metadata["training_timestamp"]).replace("Z", "+00:00"))
    if trained_at.tzinfo is None:
        raise ValueError("Training timestamp must include timezone")
    for key in ("model_name", "model_version", "review_notes", "validated_region"):
        if not isinstance(metadata[key], str) or not metadata[key].strip():
            raise ValueError(f"Invalid artifact identity field: {key}")
    for key in ("dataset_fingerprint", "artifact_sha256"):
        value = metadata[key]
        if not isinstance(value, str) or len(value) != 64:
            raise ValueError(f"Invalid SHA-256 field: {key}")
        int(value, 16)
    if metadata["task"] == "road_segment_disruption":
        if metadata["target_horizon_hours"] != 6:
            raise ValueError("Initial disruption artifact horizon must be six hours")
        threshold = metadata["selected_threshold"]
        if not isinstance(threshold, (int, float)) or not 0 <= threshold <= 1:
            raise ValueError("Invalid disruption threshold")
    if not isinstance(metadata["serving_approved"], bool):
        raise ValueError("serving_approved must be boolean")
    if not isinstance(metadata["metrics"], dict) or not metadata["metrics"]:
        raise ValueError("Artifact metrics are required")
    for name, value in metadata["metrics"].items():
        if isinstance(value, (int, float)) and (not math.isfinite(value) or value < 0):
            raise ValueError(f"Invalid artifact metric: {name}")
    provenance = metadata["dataset_provenance"]
    if provenance.get("kind") in {"synthetic", "test_fixture", "research_external"}:
        if metadata["serving_approved"]:
            raise ValueError("Synthetic/external research artifacts cannot be approved")
    if provenance.get("research_id") == "nasa_kentucky":
        if metadata["serving_approved"] or metadata["validated_region"] != "kentucky_research_only":
            raise ValueError("Kentucky artifact cannot serve as a Northeast India model")
    if metadata["serving_approved"]:
        if provenance.get("kind") != "real_ner_labels":
            raise ValueError("Approved V2 artifacts require real NER labels")
        if metadata["validated_region"] == "not_regionally_validated":
            raise ValueError("Approved artifact requires regional validation")
    if require_approved and metadata["serving_approved"] is not True:
        raise ValueError("Production serving rejects unapproved artifacts")


def save_candidate(
    estimator,
    output: Path,
    *,
    model_name: str,
    model_version: str,
    task: str,
    dataset_manifest: dict,
    validation: dict,
    input_support: dict,
) -> tuple[Path, Path]:
    output.mkdir(parents=True, exist_ok=False)
    artifact = output / "model.joblib"
    joblib.dump(estimator, artifact, compress=3)
    versions = {"python": platform.python_version()}
    for package in ("numpy", "pandas", "scikit-learn", "joblib", "xgboost", "lightgbm", "catboost"):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            pass
    metadata = {
        "schema_version": "2",
        "model_name": model_name,
        "model_version": model_version,
        "task": task,
        "feature_version": FEATURE_VERSION,
        "training_timestamp": datetime.now(timezone.utc).isoformat(),
        "dataset_fingerprint": dataset_manifest["dataset_fingerprint_sha256"],
        "dataset_provenance": {
            "kind": dataset_manifest["kind"],
            "sources": dataset_manifest["sources"],
            "licenses": dataset_manifest["licenses"],
        },
        "training_geography": dataset_manifest["geographic_coverage"],
        "validation_geography": validation["validation_geography"],
        "temporal_coverage": dataset_manifest["temporal_coverage"],
        "target_horizon_hours": validation.get("target_horizon_hours"),
        "selected_threshold": validation.get("selected_threshold"),
        "calibration_method": validation.get("calibration_method", "none"),
        "metrics": validation["metrics"],
        "library_versions": versions,
        "artifact_sha256": hashlib.sha256(artifact.read_bytes()).hexdigest(),
        "serving_approved": False,
        "review_notes": "Candidate only; independent review and approval required.",
        "validated_region": validation.get("validated_region", "not_regionally_validated"),
        "input_support": input_support,
    }
    validate_metadata(metadata)
    metadata_path = output / "model.metadata.json"
    with metadata_path.open("x", encoding="utf-8") as stream:
        json.dump(metadata, stream, indent=2, allow_nan=False, sort_keys=True)
    return artifact, metadata_path


def verify_artifact(artifact: Path, metadata_path: Path, *, require_approved=True) -> dict:
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    validate_metadata(metadata, require_approved=require_approved)
    if hashlib.sha256(artifact.read_bytes()).hexdigest() != metadata["artifact_sha256"]:
        raise ValueError("Artifact SHA-256 mismatch")
    return metadata
