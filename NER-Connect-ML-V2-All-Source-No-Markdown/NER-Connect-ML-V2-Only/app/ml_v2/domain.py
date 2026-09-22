"""Conservative input-domain and feature-support validation (not full OOD detection)."""

from dataclasses import dataclass

from app.ml_v2.contract import REQUIRED_DISRUPTION_FEATURES
from app.ml_v2.schemas import SegmentObservation

NER_STATES = {
    "arunachal pradesh",
    "assam",
    "manipur",
    "meghalaya",
    "mizoram",
    "nagaland",
    "sikkim",
    "tripura",
}


@dataclass(frozen=True)
class SupportResult:
    supported: bool
    missing_features: tuple[str, ...]
    warnings: tuple[str, ...]


def validate_feature_support(
    segment: SegmentObservation,
    *,
    supported_states: set[str] | None = None,
    supported_categories: dict[str, set[str]] | None = None,
) -> SupportResult:
    states = {state.casefold() for state in (supported_states or NER_STATES)}
    warnings = []
    missing = tuple(
        name for name in REQUIRED_DISRUPTION_FEATURES if getattr(segment, name, None) is None
    )
    if segment.state.casefold() not in states:
        warnings.append("Unsupported geography; approved ML output is prohibited.")
    for name, allowed in (supported_categories or {}).items():
        value = getattr(segment, name, None)
        if value is not None and value not in allowed:
            warnings.append(f"Unsupported categorical value for {name}.")
    if missing:
        warnings.append("Required ML features are missing; values were not replaced with zero.")
    return SupportResult(not warnings, missing, tuple(warnings))
