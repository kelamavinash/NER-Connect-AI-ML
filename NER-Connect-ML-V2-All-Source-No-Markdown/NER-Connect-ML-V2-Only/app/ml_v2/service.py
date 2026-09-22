"""V2 inference boundary with approved-ML gating and controlled fallback."""

from typing import Protocol

from app.ml_v2.contract import DISRUPTION_FEATURE_ORDER, FEATURE_VERSION
from app.ml_v2.domain import validate_feature_support
from app.ml_v2.heuristics import heuristic_intelligence
from app.ml_v2.schemas import IntelligenceRequest, IntelligenceResponse, SegmentIntelligence


class ApprovedDisruptionModel(Protocol):
    metadata: dict

    def predict_segment(self, features: dict) -> tuple[float, dict | str, list[str]]: ...


class IntelligenceServiceV2:
    def __init__(self, model: ApprovedDisruptionModel | None = None):
        self.model = model

    def _model_is_eligible(self, segment) -> tuple[bool, list[str]]:
        if self.model is None:
            return False, ["No approved Northeast India disruption model is configured."]
        metadata = self.model.metadata
        reasons = []
        if metadata.get("serving_approved") is not True:
            reasons.append("Configured artifact is not serving-approved.")
        if metadata.get("feature_version") != FEATURE_VERSION:
            reasons.append("Configured artifact feature contract is incompatible.")
        if metadata.get("task") != "road_segment_disruption":
            reasons.append("Configured artifact has the wrong task identity.")
        if metadata.get("dataset_provenance", {}).get("kind") != "real_ner_labels":
            reasons.append("Configured artifact lacks real Northeast India label provenance.")
        support = validate_feature_support(
            segment,
            supported_states=set(metadata.get("input_support", {}).get("states", [])) or None,
            supported_categories={
                key: set(values)
                for key, values in metadata.get("input_support", {}).get("categories", {}).items()
            },
        )
        reasons.extend(support.warnings)
        return not reasons, reasons

    def _live_ml(self, segment) -> SegmentIntelligence:
        features = {
            "state": segment.state,
            "district": segment.district,
            **{name: getattr(segment, name) for name in DISRUPTION_FEATURE_ORDER},
        }
        probability, uncertainty, contributors = self.model.predict_segment(features)
        if not 0 <= probability <= 1:
            raise ValueError("Nonfinite or out-of-range disruption prediction")
        fallback = heuristic_intelligence(segment, [])
        metadata = self.model.metadata
        return fallback.model_copy(
            update={
                "availability": "live_ml",
                "method": "ml",
                "disruption_probability": probability,
                "model_uncertainty": uncertainty,
                "model_version": metadata["model_version"],
                "validated_region": metadata["validated_region"],
                "warnings": [],
                "top_contributing_features": contributors,
            }
        )

    def analyze(self, request: IntelligenceRequest) -> IntelligenceResponse:
        results = []
        for segment in request.segments:
            eligible, warnings = self._model_is_eligible(segment)
            if eligible:
                try:
                    results.append(self._live_ml(segment))
                    continue
                except Exception:
                    warnings.append("ML runtime failure; controlled heuristic fallback used.")
            results.append(heuristic_intelligence(segment, warnings))
        if len(results) != len(request.segments):
            raise RuntimeError("Segment intelligence count mismatch")
        return IntelligenceResponse(route_id=request.route_id, segments=results)
