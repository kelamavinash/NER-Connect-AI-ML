"""Comparable classifier/regressor builders with optional native CatBoost categoricals."""

from dataclasses import dataclass

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from training.ml_v2.config import ExperimentConfig, effective_device


@dataclass(frozen=True)
class ModelCandidate:
    estimator: object
    uses_native_categoricals: bool
    device: str
    device_warning: str | None = None


def _preprocessor(numeric: list[str], categorical: list[str]) -> ColumnTransformer:
    return ColumnTransformer(
        [
            (
                "numeric",
                make_pipeline(
                    SimpleImputer(
                        strategy="median", add_indicator=True, keep_empty_features=True
                    ),
                    StandardScaler(),
                ),
                numeric,
            ),
            (
                "categorical",
                make_pipeline(
                    SimpleImputer(strategy="constant", fill_value="__MISSING__"),
                    OneHotEncoder(handle_unknown="ignore"),
                ),
                categorical,
            ),
        ]
    )


def classifier_candidate(
    name: str, config: ExperimentConfig, numeric: list[str], categorical: list[str], *, force_cpu=False
) -> ModelCandidate:
    requested, warning = effective_device(config)
    device = "cpu" if force_cpu else requested
    seed = config.seed
    if name == "catboost":
        from catboost import CatBoostClassifier

        estimator = CatBoostClassifier(
            iterations=250,
            depth=6,
            learning_rate=0.05,
            loss_function="Logloss",
            eval_metric="PRAUC",
            auto_class_weights="Balanced",
            random_seed=seed,
            verbose=False,
            task_type="GPU" if device == "cuda" else "CPU",
            cat_features=categorical,
        )
        return ModelCandidate(estimator, True, device, warning)
    if name == "logistic_regression":
        core = LogisticRegression(class_weight="balanced", max_iter=2000, random_state=seed)
    elif name == "random_forest":
        core = RandomForestClassifier(
            n_estimators=250, class_weight="balanced", n_jobs=1, random_state=seed
        )
    elif name == "extra_trees":
        core = ExtraTreesClassifier(
            n_estimators=250, class_weight="balanced", n_jobs=1, random_state=seed
        )
    elif name == "xgboost":
        from xgboost import XGBClassifier

        core = XGBClassifier(
            n_estimators=250,
            tree_method="hist",
            device="cuda" if device == "cuda" else "cpu",
            eval_metric="logloss",
            n_jobs=1,
            random_state=seed,
        )
    elif name == "lightgbm":
        from lightgbm import LGBMClassifier

        core = LGBMClassifier(
            n_estimators=250,
            class_weight="balanced",
            device_type="gpu" if device == "cuda" else "cpu",
            n_jobs=1,
            random_state=seed,
            verbosity=-1,
        )
    else:
        raise ValueError(f"Unknown classifier: {name}")
    return ModelCandidate(make_pipeline(_preprocessor(numeric, categorical), core), False, device, warning)


def delay_regressor(name: str, config: ExperimentConfig, numeric: list[str], categorical: list[str]):
    requested, warning = effective_device(config)
    if name == "catboost":
        from catboost import CatBoostRegressor

        return ModelCandidate(
            CatBoostRegressor(
                iterations=250,
                depth=6,
                loss_function="MAE",
                random_seed=config.seed,
                verbose=False,
                task_type="GPU" if requested == "cuda" else "CPU",
                cat_features=categorical,
            ),
            True,
            requested,
            warning,
        )
    if name == "xgboost":
        from xgboost import XGBRegressor

        core = XGBRegressor(
            n_estimators=250,
            tree_method="hist",
            device="cuda" if requested == "cuda" else "cpu",
            n_jobs=1,
            random_state=config.seed,
        )
    elif name == "lightgbm":
        from lightgbm import LGBMRegressor

        core = LGBMRegressor(
            n_estimators=250,
            device_type="gpu" if requested == "cuda" else "cpu",
            n_jobs=1,
            random_state=config.seed,
            verbosity=-1,
        )
    else:
        raise ValueError(f"Unknown delay regressor: {name}")
    return ModelCandidate(make_pipeline(_preprocessor(numeric, categorical), core), False, requested, warning)
