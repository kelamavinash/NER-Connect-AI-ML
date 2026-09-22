"""Optional, non-causal tree explanation infrastructure."""

import hashlib
from dataclasses import dataclass


@dataclass(frozen=True)
class ExplanationBatch:
    status: str
    method: str
    feature_order: tuple[str, ...]
    background_fingerprint: str | None
    top_contributors: tuple[tuple[str, ...], ...]
    warning: str


def unavailable_explanations(feature_order: tuple[str, ...], rows: int) -> ExplanationBatch:
    return ExplanationBatch(
        status="unavailable",
        method="none",
        feature_order=feature_order,
        background_fingerprint=None,
        top_contributors=tuple(() for _ in range(rows)),
        warning="SHAP was not computed; no feature-contribution claim is available.",
    )


def shap_tree_explanations(
    estimator,
    frame,
    background,
    feature_order: tuple[str, ...],
    *,
    top_k: int = 3,
    max_rows: int = 256,
) -> ExplanationBatch:
    """Compute SHAP only when explicitly requested with a reviewed background set."""
    import numpy as np

    try:
        import shap
    except ImportError as exc:
        raise RuntimeError("Install the optional SHAP dependency to compute explanations") from exc
    if not feature_order or list(frame.columns) != list(feature_order):
        raise ValueError("Explanation frame must preserve the artifact feature order")
    if list(background.columns) != list(feature_order) or not 1 <= len(background) <= 1000:
        raise ValueError("A bounded, feature-ordered background dataset is required")
    if not 1 <= len(frame) <= max_rows or not 1 <= top_k <= len(feature_order):
        raise ValueError("Explanation batch or top_k exceeds configured latency limits")
    fingerprint = hashlib.sha256(
        background.to_csv(index=False).encode("utf-8")
    ).hexdigest()
    explainer = shap.TreeExplainer(estimator, data=background)
    values = np.asarray(explainer.shap_values(frame))
    if values.ndim == 3:
        values = values[:, :, -1]
    if values.shape != (len(frame), len(feature_order)) or not np.isfinite(values).all():
        raise ValueError("Unexpected or nonfinite SHAP output")
    contributors = tuple(
        tuple(feature_order[index] for index in np.argsort(np.abs(row))[-top_k:][::-1])
        for row in values
    )
    return ExplanationBatch(
        status="computed",
        method="shap_tree",
        feature_order=feature_order,
        background_fingerprint=fingerprint,
        top_contributors=contributors,
        warning="Feature contributions describe this model output and are not causal claims.",
    )
