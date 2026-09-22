"""Delay task scaffold that refuses to train without verified delay labels."""

from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class DelayTrainingStatus:
    status: str
    reason: str
    candidate_models: tuple[str, ...] = ("lightgbm", "catboost", "xgboost")
    target: str = "delay_minutes"


def assess_delay_labels(rows: list[dict]) -> DelayTrainingStatus:
    labeled = [row for row in rows if row.get("delay_minutes") is not None]
    if not labeled:
        return DelayTrainingStatus(
            "unavailable_data", "No verified delay_minutes labels are present; training prohibited."
        )
    if any(row.get("delay_label_verified") is not True for row in labeled):
        return DelayTrainingStatus(
            "unavailable_data", "Delay values lack explicit verified-label provenance."
        )
    if len(labeled) < 100:
        return DelayTrainingStatus(
            "unavailable_data", "Insufficient verified labels for four-way validation."
        )
    return DelayTrainingStatus("pipeline_ready", "Verified labels passed initial availability checks.")


def require_delay_training_data(rows: list[dict]) -> None:
    status = assess_delay_labels(rows)
    if status.status != "pipeline_ready":
        raise ValueError(f"Delay model {status.status}: {status.reason}")


def delay_training_metadata(rows: list[dict]) -> dict:
    """Serializable interface status emitted before any delay training attempt."""
    status = assess_delay_labels(rows)
    return {
        **asdict(status),
        "metrics": ["mae", "rmse", "median_absolute_error", "p90_absolute_error"],
        "required_breakdowns": ["district", "road_class", "weather_severity", "season"],
        "vehicle_cargo_breakdown": "only_if_present_in_verified_label_generation",
        "serving_approved": False,
    }
