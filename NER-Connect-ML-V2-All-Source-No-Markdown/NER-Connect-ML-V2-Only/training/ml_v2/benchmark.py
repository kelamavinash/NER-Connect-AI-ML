"""Road-segment disruption benchmark; selection/calibration/test remain isolated."""

import csv
import importlib.util
import io
import json
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.frozen import FrozenEstimator
from sklearn.metrics import f1_score

from app.ml_v2.contract import (
    CATEGORICAL_FEATURES,
    DISRUPTION_FEATURE_ORDER,
    REQUIRED_DISRUPTION_FEATURES,
)
from training.evaluate import classification_metrics
from training.ml_v2.artifacts import save_candidate
from training.ml_v2.config import ExperimentConfig
from training.ml_v2.models import classifier_candidate
from training.ml_v2.runs import RunDirectory
from training.ml_v2.schema import TrainingTable
from training.ml_v2.splits import record_split_summary, validate_spatial_buffer

MODEL_FEATURES = ("state", "district", *DISRUPTION_FEATURE_ORDER)


def _available_candidates(names: tuple[str, ...]) -> tuple[list[str], dict[str, str]]:
    packages = {"xgboost": "xgboost", "lightgbm": "lightgbm", "catboost": "catboost"}
    available, skipped = [], {}
    for name in names:
        package = packages.get(name)
        if package and importlib.util.find_spec(package) is None:
            skipped[name] = "Optional dependency is not installed."
        else:
            available.append(name)
    return available, skipped


def _frames(table: TrainingTable):
    frame = pd.DataFrame(table.rows)
    target = "disruption_next_6h"
    if frame[target].isna().any() or set(frame[target].unique()) != {0, 1}:
        raise ValueError("Benchmark requires explicit positive and documented-control labels")
    missing_columns = [name for name in MODEL_FEATURES if name not in frame]
    if missing_columns:
        raise ValueError(f"Training table lacks V2 feature columns: {missing_columns}")
    missing_required = [name for name in REQUIRED_DISRUPTION_FEATURES if frame[name].isna().any()]
    if missing_required:
        raise ValueError(f"Required ML features contain missing values: {missing_required}")
    categorical = [name for name in MODEL_FEATURES if name in CATEGORICAL_FEATURES]
    numeric = [name for name in MODEL_FEATURES if name not in categorical]
    return frame, target, numeric, categorical


def _threshold(labels, probabilities) -> float:
    choices = np.linspace(0.01, 0.99, 99)
    # Deterministic and selection-only. Higher threshold wins exact ties to reduce alert volume.
    return float(max(choices, key=lambda value: (f1_score(labels, probabilities >= value), value)))


def _fit_candidate(name, config, numeric, categorical, x_train, y_train, x_cal, y_cal):
    candidate = classifier_candidate(name, config, numeric, categorical)
    device_warning = candidate.device_warning
    inputs = x_train.copy()
    calibration = x_cal.copy()
    if candidate.uses_native_categoricals:
        for column in categorical:
            inputs[column] = inputs[column].fillna("__MISSING__").astype(str)
            calibration[column] = calibration[column].fillna("__MISSING__").astype(str)
    try:
        candidate.estimator.fit(inputs, y_train)
    except Exception:
        if candidate.device != "cuda" or not config.allow_device_fallback:
            raise
        candidate = classifier_candidate(name, config, numeric, categorical, force_cpu=True)
        device_warning = f"{name} CUDA training failed; explicit CPU fallback active."
        candidate.estimator.fit(inputs, y_train)
    calibrated = CalibratedClassifierCV(
        FrozenEstimator(candidate.estimator), method=config.calibration_method, cv=2
    )
    calibrated.fit(calibration, y_cal)
    return calibrated, candidate.estimator, device_warning, candidate.device


def run_disruption_benchmark(
    table: TrainingTable,
    runs_root: Path,
    config: ExperimentConfig,
    *,
    run_id: str | None = None,
    require_all: bool = True,
    expected_dataset_kind: str = "real_ner_labels",
) -> dict:
    config.validate()
    if config.task != "road_segment_disruption" or config.target_horizon_hours != 6:
        raise ValueError("This entrypoint trains only disruption_next_6h")
    if table.manifest["kind"] != expected_dataset_kind:
        raise ValueError("Dataset evidence kind does not match the requested benchmark mode")
    frame, target, numeric, categorical = _frames(table)
    split_summary = record_split_summary(
        list(table.rows),
        table.manifest["split_membership"],
        enforce_unseen_district=config.unseen_district_holdout,
    )
    if config.spatial_buffer_km is not None:
        validate_spatial_buffer(
            list(table.rows), table.manifest["split_membership"], config.spatial_buffer_km
        )
    sample_to_index = {sample: index for index, sample in enumerate(frame["sample_id"])}
    indices = {
        part: np.asarray([sample_to_index[sample] for sample in ids])
        for part, ids in table.manifest["split_membership"].items()
    }
    x = frame[list(MODEL_FEATURES)]
    x = x.copy()
    for column in categorical:
        x[column] = x[column].fillna("__MISSING__").astype(str)
    y = frame[target].astype(int).to_numpy()
    for part, part_indices in indices.items():
        if set(y[part_indices]) != {0, 1}:
            raise ValueError(f"{part} partition requires both classes; revise sampling, not seed")
    available, skipped = _available_candidates(config.model_candidates)
    if require_all and skipped:
        raise RuntimeError(f"Required model dependencies are unavailable: {sorted(skipped)}")
    if not available:
        raise RuntimeError("No model candidates are available")
    run = RunDirectory.create(runs_root, run_id)
    run.write_json("config.json", config.to_dict())
    run.write_json("dataset-manifest.json", table.manifest)
    run.write_json("split-membership.json", split_summary)
    run.write_json(
        "data-quality.json",
        {
            "rows": len(frame),
            "class_prevalence": float(y.mean()),
            "missing_by_feature": {name: int(frame[name].isna().sum()) for name in MODEL_FEATURES},
            "storm_group_validation": (
                "performed" if any(row.get("storm_event_id") for row in table.rows) else "unavailable"
            ),
        },
    )
    train, cal, selection, test = (indices[name] for name in ("train", "calibration", "selection", "test"))
    reports, fitted = [], {}
    calibration_reports = {}
    for name in available:
        started = time.perf_counter()
        estimator, raw_estimator, warning, training_device = _fit_candidate(
            name, config, numeric, categorical, x.iloc[train], y[train], x.iloc[cal], y[cal]
        )
        selection_probabilities = estimator.predict_proba(x.iloc[selection])[:, 1]
        uncalibrated_selection = raw_estimator.predict_proba(x.iloc[selection])[:, 1]
        metrics = classification_metrics(y[selection], selection_probabilities)
        reports.append(
            {
                "model": name,
                **metrics,
                "train_seconds": time.perf_counter() - started,
                "artifact_size_bytes": _serialized_size(estimator),
                "training_device": training_device,
                "device_warning": warning,
            }
        )
        fitted[name] = estimator
        calibration_reports[name] = {
            "uncalibrated_brier": classification_metrics(
                y[selection], uncalibrated_selection
            )["brier"],
            "calibrated_selection_brier": metrics["brier"],
            "selection_reliability": metrics["reliability"],
        }
    winner = sorted(reports, key=lambda row: (-row["pr_auc"], row["brier"], row["model"]))[0]
    model = fitted[winner["model"]]
    selection_probabilities = model.predict_proba(x.iloc[selection])[:, 1]
    threshold = _threshold(y[selection], selection_probabilities)
    test_probabilities = model.predict_proba(x.iloc[test])[:, 1]
    inference_started = time.perf_counter()
    model.predict_proba(x.iloc[test])
    inference_seconds = time.perf_counter() - inference_started
    test_metrics = classification_metrics(y[test], test_probabilities, threshold)
    test_metrics["batch_inference_seconds"] = inference_seconds
    test_metrics["inference_latency_ms_per_sample"] = 1000 * inference_seconds / len(test)
    test_metrics["batch_throughput_samples_per_second"] = len(test) / max(
        inference_seconds, 1e-12
    )
    benchmark = {
        "task": config.task,
        "target": target,
        "class_prevalence": float(y.mean()),
        "benchmarks": reports,
        "skipped": skipped,
        "selected_model": winner["model"],
        "selection_policy": "PR-AUC descending, Brier ascending; no final-test access",
        "selected_threshold": threshold,
        "final_test_metrics": test_metrics,
        "serving_approved": False,
    }
    run.write_json("benchmark.json", benchmark)
    run.write_json("calibration-report.json", calibration_reports[winner["model"]])
    prediction_path = run.path / "test-predictions.csv"
    with prediction_path.open("x", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=["sample_id", "label", "probability"])
        writer.writeheader()
        for index, probability in zip(test, test_probabilities, strict=True):
            writer.writerow(
                {
                    "sample_id": frame.iloc[index]["sample_id"],
                    "label": int(y[index]),
                    "probability": float(probability),
                }
            )
    support = {
        "states": sorted(frame.iloc[train]["state"].unique().tolist()),
        "categories": {
            name: sorted(frame.iloc[train][name].dropna().astype(str).unique().tolist())
            for name in categorical
        },
        "numeric_ranges": {
            name: [float(frame.iloc[train][name].min()), float(frame.iloc[train][name].max())]
            for name in numeric
            if frame.iloc[train][name].notna().any()
        },
    }
    artifact_dir = run.path / "artifact"
    artifact, metadata = save_candidate(
        model,
        artifact_dir,
        model_name=winner["model"],
        model_version=run.path.name,
        task=config.task,
        dataset_manifest=table.manifest,
        validation={
            "validation_geography": {part: value["districts"] for part, value in split_summary.items()},
            "target_horizon_hours": 6,
            "selected_threshold": threshold,
            "calibration_method": config.calibration_method,
            "metrics": test_metrics,
            "validated_region": "northeast_india_dataset_specific",
        },
        input_support=support,
    )
    candidate_metadata = json.loads(metadata.read_text(encoding="utf-8"))
    run.write_json("selected-model-metadata.json", candidate_metadata)
    run.write_json("sha256.json", {"artifact_sha256": candidate_metadata["artifact_sha256"]})
    run.write_json(
        "environment-versions.json", candidate_metadata["library_versions"]
    )
    return benchmark


def _serialized_size(estimator) -> int:
    stream = io.BytesIO()
    joblib.dump(estimator, stream)
    return stream.tell()
