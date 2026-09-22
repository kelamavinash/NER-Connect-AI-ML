"""Trusted local V2 artifact loader with production approval enforcement."""

import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from app.ml_v2.contract import DISRUPTION_FEATURE_ORDER, FEATURE_VERSION

SERVING_FEATURE_ORDER = ("state", "district", *DISRUPTION_FEATURE_ORDER)
MAX_ARTIFACT_BYTES = 512 * 1024 * 1024


class ArtifactV2Error(ValueError):
    pass


@dataclass(frozen=True)
class LoadedDisruptionModel:
    estimator: Any
    metadata: dict

    def predict_segment(self, features: dict) -> tuple[float, str, list[str]]:
        import numpy as np
        import pandas as pd

        row = {name: features.get(name) for name in SERVING_FEATURE_ORDER}
        frame = pd.DataFrame([row], columns=SERVING_FEATURE_ORDER)
        for name in self.metadata.get("input_support", {}).get("categories", {}):
            if name in frame:
                frame[name] = frame[name].fillna("__MISSING__").astype(str)
        probabilities = np.asarray(self.estimator.predict_proba(frame), dtype=float)
        if probabilities.shape != (1, 2) or not np.isfinite(probabilities).all():
            raise ArtifactV2Error("Invalid disruption probability output")
        probability = float(probabilities[0, 1])
        if not 0 <= probability <= 1 or not math.isclose(float(probabilities.sum()), 1, rel_tol=1e-6):
            raise ArtifactV2Error("Invalid disruption probability values")
        # SHAP is not computed by this loader; do not invent explanations.
        return probability, "unavailable", []


def load_disruption_artifact(
    artifact_path: Path, expected_version: str | None = None
) -> LoadedDisruptionModel:
    import joblib

    artifact = artifact_path.resolve(strict=True)
    metadata_path = artifact.with_name("model.metadata.json")
    if artifact.suffix != ".joblib" or artifact.stat().st_size > MAX_ARTIFACT_BYTES:
        raise ArtifactV2Error("Invalid artifact type or size")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    requirements = {
        "schema_version": "2",
        "task": "road_segment_disruption",
        "feature_version": FEATURE_VERSION,
        "serving_approved": True,
    }
    for name, expected in requirements.items():
        if metadata.get(name) != expected:
            raise ArtifactV2Error(f"Incompatible V2 artifact metadata: {name}")
    if expected_version and metadata.get("model_version") != expected_version:
        raise ArtifactV2Error("Configured disruption model version mismatch")
    provenance = metadata.get("dataset_provenance", {})
    if provenance.get("kind") != "real_ner_labels":
        raise ArtifactV2Error("Serving requires real Northeast India label provenance")
    if provenance.get("research_id") == "nasa_kentucky":
        raise ArtifactV2Error("Kentucky research artifacts cannot serve NER predictions")
    if not metadata.get("validated_region") or metadata["validated_region"] == "not_regionally_validated":
        raise ArtifactV2Error("Artifact lacks regional validation")
    checksum = hashlib.sha256(artifact.read_bytes()).hexdigest()
    if checksum != metadata.get("artifact_sha256"):
        raise ArtifactV2Error("Artifact SHA-256 mismatch")
    estimator = joblib.load(artifact)
    if not hasattr(estimator, "predict_proba"):
        raise ArtifactV2Error("Disruption artifact is not a probabilistic classifier")
    return LoadedDisruptionModel(estimator, metadata)
