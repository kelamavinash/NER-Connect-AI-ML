"""Serializable experiment configuration with explicit CPU/GPU behavior."""

import json
import os
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Literal


@dataclass(frozen=True)
class ExperimentConfig:
    seed: int = 42
    task: Literal["road_segment_disruption", "delay", "landslide", "flood"] = (
        "road_segment_disruption"
    )
    target_horizon_hours: int = 6
    feature_version: str = "2"
    model_candidates: tuple[str, ...] = (
        "logistic_regression",
        "random_forest",
        "extra_trees",
        "xgboost",
        "lightgbm",
        "catboost",
    )
    device: Literal["cpu", "cuda"] = "cpu"
    allow_device_fallback: bool = True
    spatial_block_degrees: float = 0.25
    unseen_district_holdout: bool = True
    spatial_buffer_km: float | None = None
    temporal_cutoffs: dict[str, str] = field(default_factory=dict)
    calibration_method: Literal["sigmoid", "isotonic"] = "sigmoid"
    threshold_policy: str = "selection_f1"
    storm_group_validation: Literal["required", "optional", "unavailable"] = "optional"

    def validate(self) -> None:
        if self.seed < 0 or self.target_horizon_hours not in (3, 6, 12):
            raise ValueError("Invalid seed or supported target horizon")
        if self.feature_version != "2" or not 0 < self.spatial_block_degrees <= 10:
            raise ValueError("Invalid feature version or spatial block size")
        if self.calibration_method == "isotonic" and self.task == "delay":
            raise ValueError("Classifier calibration does not apply to delay regression")

    def to_dict(self) -> dict:
        self.validate()
        value = asdict(self)
        value["model_candidates"] = list(self.model_candidates)
        return value

    @classmethod
    def from_json(cls, path: Path) -> "ExperimentConfig":
        raw = json.loads(path.read_text(encoding="utf-8"))
        if "model_candidates" in raw:
            raw["model_candidates"] = tuple(raw["model_candidates"])
        result = cls(**raw)
        result.validate()
        return result

    @classmethod
    def from_environment(cls) -> "ExperimentConfig":
        device = os.getenv("TRAIN_DEVICE", "cpu").casefold()
        if device not in {"cpu", "cuda"}:
            raise ValueError("TRAIN_DEVICE must be cpu or cuda")
        fallback = os.getenv("TRAIN_DEVICE_FALLBACK", "true").casefold() in {"1", "true", "yes"}
        return cls(device=device, allow_device_fallback=fallback)


def cuda_available() -> bool:
    try:
        import xgboost as xgb

        # Build information is safer than allocating a GPU during configuration.
        return bool(xgb.build_info().get("USE_CUDA"))
    except (ImportError, AttributeError):
        return False


def effective_device(config: ExperimentConfig) -> tuple[str, str | None]:
    if config.device == "cpu":
        return "cpu", None
    if cuda_available():
        return "cuda", None
    if config.allow_device_fallback:
        return "cpu", "CUDA requested but unavailable; explicit CPU fallback active."
    raise RuntimeError("CUDA requested but unavailable and CPU fallback is disabled")
