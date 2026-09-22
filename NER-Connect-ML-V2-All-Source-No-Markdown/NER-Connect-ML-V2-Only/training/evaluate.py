import time

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def classification_metrics(labels, probabilities, threshold: float = 0.5) -> dict:
    labels, probabilities = np.asarray(labels), np.asarray(probabilities, dtype=float)
    if labels.ndim != 1 or labels.shape != probabilities.shape or set(labels) != {0, 1}:
        raise ValueError("Metrics require matching vectors with both label classes")
    if (
        not np.isfinite(probabilities).all()
        or (probabilities < 0).any()
        or (probabilities > 1).any()
    ):
        raise ValueError("Invalid probabilities")
    if not 0 <= threshold <= 1:
        raise ValueError("Invalid decision threshold")
    prediction = probabilities >= threshold
    matrix = confusion_matrix(labels, prediction, labels=[0, 1])
    tn, fp, fn, tp = matrix.ravel()
    bins = np.linspace(0, 1, 11)
    expected_calibration_error = 0.0
    reliability = []
    for lower, upper in zip(bins[:-1], bins[1:], strict=True):
        mask = (probabilities >= lower) & (
            probabilities <= upper if upper == 1 else probabilities < upper
        )
        if mask.any():
            predicted_mean = float(probabilities[mask].mean())
            observed_rate = float(labels[mask].mean())
            weight = float(mask.mean())
            expected_calibration_error += weight * abs(predicted_mean - observed_rate)
            reliability.append(
                {
                    "lower": float(lower),
                    "upper": float(upper),
                    "count": int(mask.sum()),
                    "predicted_mean": predicted_mean,
                    "observed_rate": observed_rate,
                }
            )
    return {
        "roc_auc": float(roc_auc_score(labels, probabilities)),
        "pr_auc": float(average_precision_score(labels, probabilities)),
        "precision": float(precision_score(labels, prediction, zero_division=0)),
        "recall": float(recall_score(labels, prediction, zero_division=0)),
        "f1": float(f1_score(labels, prediction, zero_division=0)),
        "brier": float(brier_score_loss(labels, probabilities)),
        "false_positive_rate": float(fp / (fp + tn)) if fp + tn else 0.0,
        "false_negative_rate": float(fn / (fn + tp)) if fn + tp else 0.0,
        "expected_calibration_error": expected_calibration_error,
        "alert_volume": int(prediction.sum()),
        "class_prevalence": float(labels.mean()),
        "reliability": reliability,
        "confusion_matrix": matrix.tolist(),
        "threshold": threshold,
    }


def measure(estimator, features, labels, threshold: float = 0.5) -> dict:
    start = time.perf_counter()
    probabilities = estimator.predict_proba(features)[:, 1]
    latency_ms = (time.perf_counter() - start) * 1000 / len(features)
    return {
        **classification_metrics(labels, probabilities, threshold),
        "latency_ms_per_sample": latency_ms,
    }


def regression_metrics(labels, predictions) -> dict:
    from sklearn.metrics import mean_absolute_error, mean_squared_error, median_absolute_error

    labels, predictions = np.asarray(labels, dtype=float), np.asarray(predictions, dtype=float)
    if labels.ndim != 1 or labels.shape != predictions.shape or not len(labels):
        raise ValueError("Regression metrics require matching nonempty vectors")
    if not np.isfinite(labels).all() or not np.isfinite(predictions).all() or (labels < 0).any():
        raise ValueError("Invalid delay labels or predictions")
    errors = np.abs(labels - predictions)
    return {
        "mae": float(mean_absolute_error(labels, predictions)),
        "rmse": float(mean_squared_error(labels, predictions) ** 0.5),
        "median_absolute_error": float(median_absolute_error(labels, predictions)),
        "p90_absolute_error": float(np.quantile(errors, 0.9)),
    }
