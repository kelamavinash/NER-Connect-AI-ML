"""Scientific safety and contract tests for ML V2; fixtures make no hazard claims."""

import hashlib
import json
import sys
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import pytest
from pydantic import ValidationError

from app.ml_v2.artifacts import ArtifactV2Error, load_disruption_artifact
from app.ml_v2.contract import (
    CATEGORICAL_FEATURES,
    DISRUPTION_FEATURE_ORDER,
    FEATURES,
    FEATURE_VERSION,
    REQUIRED_DISRUPTION_FEATURES,
)
from app.ml_v2.schemas import IntelligenceRequest, SegmentObservation
from app.ml_v2.service import IntelligenceServiceV2
from training.ml_v2.artifacts import validate_metadata
from training.ml_v2.benchmark import _available_candidates, run_disruption_benchmark
from training.ml_v2.config import ExperimentConfig, effective_device
from training.ml_v2.delay import assess_delay_labels, delay_training_metadata
from training.ml_v2.explanations import shap_tree_explanations, unavailable_explanations
from training.ml_v2.feature_pipeline import FeatureValue, ObservationBuilder, nearest_join
from training.ml_v2.manifest import validate_manifest
from training.ml_v2.models import classifier_candidate
from training.ml_v2.pipeline import (
    disruption_label,
    forecast_window,
    generate_segments,
    rainfall_windows,
)
from training.ml_v2.schema import TrainingTable, validate_row
from training.ml_v2.splits import validate_spatial_buffer, validate_split_membership

NOW = datetime(2025, 6, 1, 12, tzinfo=timezone.utc)


def segment(**changes):
    values = {
        "segment_id": "seg-1",
        "observation_time": NOW,
        "latitude": 26.1,
        "longitude": 91.7,
        "state": "Assam",
        "district": "Kamrup",
        "road_id": "NH-27",
        "road_class": "trunk",
        "slope_deg": 12.0,
        "elevation_m": 110.0,
        "rain_6h_mm": 40.0,
        "rain_24h_mm": 100.0,
        "road_condition_score": 70.0,
        "source_timestamps": {"rain_6h_mm": NOW},
    }
    values.update(changes)
    return SegmentObservation.model_validate(values)


def test_feature_contract_has_complete_provenance_and_fixed_order():
    assert FEATURE_VERSION == "2"
    assert len(FEATURES) == len({feature.name for feature in FEATURES})
    assert set(REQUIRED_DISRUPTION_FEATURES) == {
        "slope_deg",
        "elevation_m",
        "road_class",
        "rain_6h_mm",
        "rain_24h_mm",
        "road_condition_score",
    }
    for feature in FEATURES:
        assert all(
            (
                feature.source,
                feature.units,
                feature.temporal_status,
                feature.missing_policy,
                feature.transformation,
                feature.spatial_resolution,
                feature.version,
            )
        )


def test_source_timestamp_cannot_come_from_future():
    with pytest.raises(ValidationError, match="source timestamps after"):
        segment(source_timestamps={"rain_6h_mm": NOW + timedelta(seconds=1)})


def test_weather_windows_are_past_only_and_missing_is_not_zero():
    values = rainfall_windows(
        NOW,
        [(NOW - timedelta(minutes=30), 2), (NOW - timedelta(hours=2), 5)],
    )
    assert values["rain_1h_mm"] == 2
    assert values["rain_3h_mm"] == 7
    assert rainfall_windows(NOW, [])["rain_24h_mm"] is None
    with pytest.raises(ValueError, match="available by T"):
        rainfall_windows(NOW, [(NOW + timedelta(minutes=1), 1)])


def test_feature_provenance_is_auditable_and_missing_stays_null():
    builder = ObservationBuilder({"sample_id": "a", "segment_id": "s"}, NOW)
    builder.add(
        FeatureValue(
            "rain_6h_mm",
            12.0,
            "archived-weather",
            NOW,
            "mm",
            "0.1 degree grid",
            "2",
        )
    )
    row, provenance = builder.build()
    assert row["rain_6h_mm"] == 12
    assert row["rain_24h_mm"] is None
    assert provenance["rain_24h_mm"]["available"] is False
    with pytest.raises(ValueError, match="Future leakage"):
        ObservationBuilder({}, NOW).add(
            FeatureValue(
                "rain_6h_mm",
                1,
                "fixture",
                NOW + timedelta(seconds=1),
                "mm",
                "fixture grid",
                "2",
            )
        )


def test_spatial_join_requires_crs_and_is_deterministic():
    match = nearest_join(
        [
            {"distance_m": 20, "crs": "EPSG:32646", "source_id": "b"},
            {"distance_m": 10, "crs": "EPSG:32646", "source_id": "a"},
        ]
    )
    assert match["source_id"] == "a"
    with pytest.raises(ValueError, match="record CRS"):
        nearest_join([{"distance_m": 1, "source_id": "bad"}])


def test_explanations_do_not_claim_shap_when_not_computed():
    result = unavailable_explanations(("rain_24h_mm", "slope_deg"), 2)
    assert result.status == "unavailable"
    assert result.method == "none"
    assert result.top_contributors == ((), ())


def test_shap_explanation_preserves_order_and_is_consistent(monkeypatch):
    pd = pytest.importorskip("pandas")
    np = pytest.importorskip("numpy")

    class FakeExplainer:
        def __init__(self, estimator, data):
            assert list(data.columns) == ["rain_24h_mm", "slope_deg"]

        def shap_values(self, frame):
            return np.asarray([[1.0, -3.0] for _ in range(len(frame))])

    monkeypatch.setitem(sys.modules, "shap", SimpleNamespace(TreeExplainer=FakeExplainer))
    frame = pd.DataFrame([[50.0, 20.0]], columns=["rain_24h_mm", "slope_deg"])
    background = pd.DataFrame([[10.0, 5.0]], columns=frame.columns)
    result = shap_tree_explanations(
        object(), frame, background, tuple(frame.columns), top_k=2
    )
    assert result.status == "computed"
    assert result.top_contributors == (("slope_deg", "rain_24h_mm"),)
    assert result.background_fingerprint


def test_forecast_vintage_must_exist_at_observation_time():
    with pytest.raises(ValueError, match="issued after"):
        forecast_window(NOW, NOW + timedelta(seconds=1), [], 6)
    assert forecast_window(
        NOW,
        NOW - timedelta(hours=1),
        [(NOW + timedelta(hours=1), 2), (NOW + timedelta(hours=7), 100)],
        6,
    ) == 2


def test_labels_require_verified_events_or_documented_controls():
    assert disruption_label("seg-1", NOW, [], documented_control=False) is None
    assert disruption_label("seg-1", NOW, [], documented_control=True) == 0
    incident = {
        "segment_id": "seg-1",
        "occurred_at": NOW + timedelta(hours=2),
        "verified": True,
        "logistics_relevant": True,
    }
    assert disruption_label("seg-1", NOW, [incident]) == 1


def test_training_row_rejects_undocumented_negative_and_predictor_label_fields():
    row = {
        "sample_id": "sample-1",
        "segment_id": "seg-1",
        "observation_time": NOW.isoformat(),
        "latitude": "26.1",
        "longitude": "91.7",
        "state": "Assam",
        "district": "Kamrup",
        "road_id": "NH-27",
        "road_class": "trunk",
        "disruption_next_6h": "0",
        "delay_minutes": "",
    }
    with pytest.raises(ValueError, match="documented control"):
        validate_row(row)
    row["control_sampling_status"] = "documented_control"
    assert validate_row(row)["disruption_next_6h"] == 0
    row["event_label"] = "leak"
    with pytest.raises(ValueError, match="may not enter predictor"):
        validate_row(row)


def test_deterministic_segment_ids_include_crs_and_sequence():
    first = generate_segments("road-1", [(91.0, 26.0), (91.1, 26.1), (91.2, 26.2)])
    second = generate_segments("road-1", [(91.0, 26.0), (91.1, 26.1), (91.2, 26.2)])
    assert first == second
    assert len(first) == 2
    assert first[0].segment_id != first[1].segment_id


def _manifest():
    return {
        "dataset_name": "software-fixture",
        "schema_version": "2",
        "created_at": NOW.isoformat(),
        "kind": "test_fixture",
        "geographic_coverage": {"states": ["Assam"]},
        "temporal_coverage": {"start": "2025-01-01", "end": "2025-02-01"},
        "sources": ["artificial software fixture"],
        "licenses": ["test only"],
        "feature_definitions": {},
        "positive_target_definition": "test-only binary",
        "negative_control_sampling_method": "explicit fixture controls",
        "disruption_definition": "test only",
        "label_source": "fixture generator",
        "deduplication_rules": "unique sample",
        "spatial_resolution": "test segment",
        "road_segment_generation_method": "fixed fixture",
        "weather_alignment_method": "past-only fixture",
        "missing_data_policy": "null preserved",
        "dataset_fingerprint_sha256": hashlib.sha256(b"fixture").hexdigest(),
        "split_membership": {"train": ["a"], "calibration": ["b"], "selection": ["c"], "test": ["d"]},
        "known_limitations": ["not real"],
    }


def test_manifest_identity_and_split_membership_are_validated():
    assert validate_manifest(_manifest())["schema_version"] == "2"
    bad = _manifest()
    bad["split_membership"]["test"] = ["a"]
    with pytest.raises(ValueError, match="more than one"):
        validate_manifest(bad)


def _split_rows():
    rows = []
    for index, part in enumerate(("train", "calibration", "selection", "test")):
        rows.append(
            {
                "sample_id": part,
                "observation_time": NOW + timedelta(days=index),
                "spatial_group": f"g-{index}",
                "storm_event_id": f"storm-{index}",
                "district": f"district-{index}",
            }
        )
    return rows, {part: [part] for part in ("train", "calibration", "selection", "test")}


def test_spatial_temporal_storm_and_district_leakage_guards():
    rows, membership = _split_rows()
    validate_split_membership(rows, membership, enforce_unseen_district=True)
    rows[1]["spatial_group"] = rows[0]["spatial_group"]
    with pytest.raises(ValueError, match="spatial_group leaks"):
        validate_split_membership(rows, membership)


def test_temporal_order_is_not_reshuffled_for_results():
    rows, membership = _split_rows()
    rows[-1]["observation_time"] = NOW - timedelta(days=1)
    with pytest.raises(ValueError, match="progress"):
        validate_split_membership(rows, membership)


def test_spatial_buffer_holdout_rejects_nearby_final_test():
    rows, membership = _split_rows()
    for index, row in enumerate(rows):
        row["latitude"] = 26.0 + index
        row["longitude"] = 91.0
    validate_spatial_buffer(rows, membership, 10)
    rows[-1]["latitude"] = 26.001
    with pytest.raises(ValueError, match="spatial buffer"):
        validate_spatial_buffer(rows, membership, 10)


def test_delay_model_stays_unavailable_without_verified_labels():
    assert assess_delay_labels([{"delay_minutes": None}]).status == "unavailable_data"
    assert assess_delay_labels([{"delay_minutes": 12}]).status == "unavailable_data"
    metadata = delay_training_metadata([{"delay_minutes": None}])
    assert metadata["status"] == "unavailable_data"
    assert metadata["serving_approved"] is False
    assert "p90_absolute_error" in metadata["metrics"]


def _artifact_metadata(kind="real_ner_labels", approved=False):
    return {
        "model_name": "fixture",
        "model_version": "v1",
        "task": "road_segment_disruption",
        "feature_version": "2",
        "training_timestamp": NOW.isoformat(),
        "dataset_fingerprint": "0" * 64,
        "dataset_provenance": {"kind": kind},
        "training_geography": {},
        "validation_geography": {},
        "temporal_coverage": {},
        "target_horizon_hours": 6,
        "selected_threshold": 0.5,
        "calibration_method": "sigmoid",
        "metrics": {"pr_auc": 0.5},
        "library_versions": {},
        "artifact_sha256": "0" * 64,
        "serving_approved": approved,
        "review_notes": "fixture",
        "validated_region": "northeast_india_dataset_specific",
        "input_support": {},
    }


def test_artifact_approval_synthetic_and_kentucky_guards():
    with pytest.raises(ValueError, match="unapproved"):
        validate_metadata(_artifact_metadata(), require_approved=True)
    with pytest.raises(ValueError, match="cannot be approved"):
        validate_metadata(_artifact_metadata("test_fixture", approved=True))
    kentucky = _artifact_metadata("research_external", approved=True)
    kentucky["dataset_provenance"]["research_id"] = "nasa_kentucky"
    with pytest.raises(ValueError):
        validate_metadata(kentucky)


def test_unapproved_artifact_is_rejected_before_deserialization(tmp_path, monkeypatch):
    artifact = tmp_path / "model.joblib"
    artifact.write_bytes(b"not-a-pickle")
    metadata = _artifact_metadata(approved=False)
    metadata["schema_version"] = "2"
    metadata["artifact_sha256"] = hashlib.sha256(artifact.read_bytes()).hexdigest()
    (tmp_path / "model.metadata.json").write_text(json.dumps(metadata), encoding="utf-8")
    called = False

    def forbidden(*args, **kwargs):
        nonlocal called
        called = True
        raise AssertionError("unapproved artifact reached deserialization")

    monkeypatch.setattr("joblib.load", forbidden)
    with pytest.raises(ArtifactV2Error, match="serving_approved"):
        load_disruption_artifact(artifact)
    assert called is False


def test_cpu_config_and_explicit_cuda_fallback(monkeypatch):
    assert effective_device(ExperimentConfig(device="cpu")) == ("cpu", None)
    monkeypatch.setattr("training.ml_v2.config.cuda_available", lambda: False)
    device, warning = effective_device(ExperimentConfig(device="cuda"))
    assert device == "cpu" and "fallback" in warning
    with pytest.raises(RuntimeError):
        effective_device(ExperimentConfig(device="cuda", allow_device_fallback=False))


def test_catboost_is_optional_and_reported_when_missing(monkeypatch):
    original = __import__("importlib.util").util.find_spec
    monkeypatch.setattr(
        "training.ml_v2.benchmark.importlib.util.find_spec",
        lambda name: None if name == "catboost" else original(name),
    )
    available, skipped = _available_candidates(("logistic_regression", "catboost"))
    assert available == ["logistic_regression"]
    assert "catboost" in skipped


def test_catboost_keeps_categoricals_native_when_installed():
    pytest.importorskip("catboost")
    candidate = classifier_candidate(
        "catboost",
        ExperimentConfig(model_candidates=("catboost",)),
        ["slope_deg"],
        ["state", "road_class"],
    )
    assert candidate.uses_native_categoricals is True
    assert candidate.estimator.get_params()["cat_features"] == ["state", "road_class"]


def test_unavailable_model_returns_honest_heuristic_metadata():
    request = IntelligenceRequest(route_id="route-1", segments=[segment()])
    response = IntelligenceServiceV2().analyze(request)
    assert len(response.segments) == 1
    result = response.segments[0]
    assert result.method == "heuristic"
    assert result.disruption_probability is None
    assert result.model_uncertainty == "unavailable"
    assert result.predicted_delay_minutes is None


def test_unsupported_geography_cannot_silently_serve_ml():
    class Model:
        metadata = {
            "serving_approved": True,
            "feature_version": "2",
            "task": "road_segment_disruption",
            "dataset_provenance": {"kind": "real_ner_labels"},
            "input_support": {"states": ["Assam"]},
            "model_version": "v2",
            "validated_region": "Assam",
        }

        def predict_segment(self, features):
            raise AssertionError("unsupported geography must not call model")

    request = IntelligenceRequest(route_id="route-1", segments=[segment(state="Bihar")])
    result = IntelligenceServiceV2(Model()).analyze(request).segments[0]
    assert result.method == "heuristic"
    assert any("Unsupported geography" in warning for warning in result.warnings)


def test_missing_required_feature_cannot_become_zero_risk_ml_output():
    class Model:
        metadata = {
            "serving_approved": True,
            "feature_version": "2",
            "task": "road_segment_disruption",
            "dataset_provenance": {"kind": "real_ner_labels"},
            "input_support": {"states": ["Assam"]},
            "model_version": "v2",
            "validated_region": "Assam",
        }

    request = IntelligenceRequest(route_id="route-1", segments=[segment(rain_24h_mm=None)])
    result = IntelligenceServiceV2(Model()).analyze(request).segments[0]
    assert result.disruption_probability is None
    assert "rain_24h_mm" in result.data_quality.missing_features


def test_nonfinite_prediction_uses_controlled_fallback():
    class Model:
        metadata = {
            "serving_approved": True,
            "feature_version": "2",
            "task": "road_segment_disruption",
            "dataset_provenance": {"kind": "real_ner_labels"},
            "input_support": {"states": ["Assam"]},
            "model_version": "v2",
            "validated_region": "Assam",
        }

        def predict_segment(self, features):
            return float("nan"), "unavailable", []

    request = IntelligenceRequest(route_id="route-1", segments=[segment()])
    result = IntelligenceServiceV2(Model()).analyze(request).segments[0]
    assert result.method == "heuristic"
    assert result.disruption_probability is None
    assert any("runtime failure" in warning for warning in result.warnings)


@pytest.mark.training
def test_disruption_pipeline_uses_artificial_fixture_without_approval(tmp_path):
    rows = []
    membership = {part: [] for part in ("train", "calibration", "selection", "test")}
    for part_index, part in enumerate(membership):
        for offset in range(8):
            sample_id = f"{part}-{offset}"
            membership[part].append(sample_id)
            row = {
                "sample_id": sample_id,
                "segment_id": f"segment-{part_index}-{offset}",
                "observation_time": NOW + timedelta(days=part_index, minutes=offset),
                "latitude": 24.0 + part_index + offset / 100,
                "longitude": 90.0 + part_index,
                "state": "Assam",
                "district": f"district-{part_index}",
                "spatial_group": f"group-{part_index}-{offset}",
                "storm_event_id": f"storm-{part_index}-{offset}",
                "disruption_next_6h": offset % 2,
                "delay_minutes": None,
            }
            for feature in DISRUPTION_FEATURE_ORDER:
                if feature in CATEGORICAL_FEATURES:
                    row[feature] = "fixture-category"
                elif feature == "road_class":
                    row[feature] = "trunk"
                elif feature == "road_condition_score":
                    row[feature] = 70.0 - offset
                elif feature == "slope_deg":
                    row[feature] = 5.0 + offset
                elif feature == "elevation_m":
                    row[feature] = 100.0 + offset
                elif feature in ("rain_6h_mm", "rain_24h_mm"):
                    row[feature] = 10.0 + 20 * (offset % 2)
                else:
                    row[feature] = float(offset)
            rows.append(row)
    manifest = _manifest()
    manifest["split_membership"] = membership
    table = TrainingTable(tuple(rows), manifest)
    result = run_disruption_benchmark(
        table,
        tmp_path,
        ExperimentConfig(
            model_candidates=("logistic_regression",), unseen_district_holdout=False
        ),
        run_id="fixture-run",
        require_all=False,
        expected_dataset_kind="test_fixture",
    )
    assert result["selected_model"] == "logistic_regression"
    assert result["serving_approved"] is False
    assert (tmp_path / "fixture-run" / "artifact" / "model.joblib").is_file()
    assert (tmp_path / "fixture-run" / "test-predictions.csv").is_file()
