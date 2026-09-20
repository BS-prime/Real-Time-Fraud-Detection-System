"""
Model training utilities.
This module provides functionality to train fraud detection models using engineered features.
It supports multiple model types (e.g., Random Forest, XGBoost) and allows hyperparameter tuning via grid search.
The trained models are saved to a specified directory for later use in inference or evaluation.
"""

import importlib
import inspect
import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
import yaml
from sklearn.model_selection import GridSearchCV, StratifiedKFold

from fraud_detection.feature_schema import TARGET_COLUMN
from fraud_detection.mlflow_tracking import MLflowTracker
from fraud_detection.model_io import ModelIO
from fraud_detection.paths import (
    HYPERPARAMS_CONFIG_PATH,
    MODEL_DIR,
    PARAMS_CONFIG_PATH,
    SAVED_MODELS_PATH,
    create_dir,
    train_feature_file_path,
)

logger = logging.getLogger(__name__)

# Modules to search for model classes when resolving model types from strings. This list includes common scikit-learn modules and popular gradient boosting libraries. If a model type is not found in these modules, it will be considered unsupported.
_MODEL_TYPE_SEARCH_MODULES = [
    "sklearn.ensemble",
    "sklearn.linear_model",
    "sklearn.tree",
    "sklearn.svm",
    "sklearn.naive_bayes",
    "sklearn.neighbors",
    "xgboost",
    "lightgbm",
    "catboost",
]

# Required methods for a class to be considered a valid estimator. This is used to filter out non-estimator classes when resolving model types from strings.
_REQUIRED_ESTIMATOR_METHODS = ("fit", "predict")

# Cache of resolved model classes to avoid repeated imports and inspections. This improves performance when training multiple models of the same type.
_model_class_cache: dict[str, type] = {}

# Hyperparameters that request internal parallelism (e.g., n_jobs) are overridden to a single job to avoid CPU oversubscription when GridSearchCV is also parallelizing across folds and candidates. This is a common source of performance degradation and confusing errors, so we enforce it here rather than relying on the user to remember to do it in config.
_ESTIMATOR_PARALLELISM_PARAMS = {"n_jobs"}

# Valid model names must start with a letter and contain only letters, digits, underscores, and hyphens. This regex pattern is used to validate model names in the configuration file to ensure they are safe to use as filenames and identifiers.
_VALID_ALGO_NAME_PATTERN = re.compile(r"^[A-Za-z][A-Za-z0-9_-]*$")


@dataclass
class TrainingResult:
    """
    Serializable result for one model training attempt.
    """

    algo_name: str
    model_path: Path | None
    best_score: float | None
    best_params: dict[str, Any]
    error: str | None = None

    @property
    def succeeded(self) -> bool:
        return self.error is None and self.model_path is not None

    def to_model_paths_value(self) -> str:
        return str(self.model_path) if self.succeeded else str(self.error)


def _resolve_model_class(model_type: str) -> type:
    """
    import the class corresponding to the given model_type string from the allow-listed modules.
    If the class is not found or does not meet the required interface, raise a ValueError
    """

    # 1. Check the cache first to avoid repeated imports and inspections.
    if model_type in _model_class_cache:
        return _model_class_cache[model_type]

    # 2. Search the allow-listed modules for a class with the given name.
    for module_name in _MODEL_TYPE_SEARCH_MODULES:
        # a. Import the module. If the module isn't installed (e.g., lightgbm), skip it.
        try:
            module = importlib.import_module(module_name)
        except ImportError:
            # skip the model if it's not installed(e.g., lightgbm)
            continue

        # b. Check if the module has an attribute with the given model_type name.
        candidate = getattr(module, model_type, None)
        if candidate is None:
            continue
        if not inspect.isclass(candidate):
            continue
        if not all(
            hasattr(candidate, method) for method in _REQUIRED_ESTIMATOR_METHODS
        ):
            logger.warning(
                "Found '%s' in %s but it doesn't look like an estimator "
                "(missing fit/predict); skipping.",
                model_type,
                module_name,
            )
            continue

        _model_class_cache[model_type] = candidate
        return candidate

    searched = ", ".join(_MODEL_TYPE_SEARCH_MODULES)

    raise ValueError(
        f"Unsupported model type '{model_type}'. Could not find a matching "
        f"estimator class in any of: {searched}. If it's in an installed "
        f"library not on this list, add its module to "
        f"_MODEL_TYPE_SEARCH_MODULES."
    )


class ModelTrainer:
    """
    Train fraud detection models using configured hyperparameters in YAML file.
    """

    def __init__(
        self,
        train_file_path: Path | None = None,
        hyperparams_config_path: Path = HYPERPARAMS_CONFIG_PATH,
        params_config_path: Path = PARAMS_CONFIG_PATH,
        model_dir: Path = MODEL_DIR,
        ) -> None:
        self.hyperparams_config_path = hyperparams_config_path
        self.param_path = params_config_path
        self.model_dir = model_dir
        self.params_config = self._load_params_config()
        self.hyperparams_config = self._load_hyperparams_config()
        self.seed = self.params_config["seed"]
        self.train_file_path = train_file_path or train_feature_file_path(
            seed=self.seed
        )
        self._validate_algo_names()

    def _load_hyperparams_config(self) -> dict[str, Any]:
        """
        Read the model and hyperparameter configuration from YAML.
        """

        if not self.hyperparams_config_path.exists():
            raise FileNotFoundError(
                f"Training configuration not found: {self.hyperparams_config_path}"
            )

        with self.hyperparams_config_path.open("r", encoding="utf-8") as file:
            return yaml.safe_load(file)

    def _load_params_config(self) -> dict[str, Any]:
        """
        Read the model training parameters from YAML configuration.
        """

        if not self.param_path.exists():
            raise FileNotFoundError(
                f"Training parameters configuration rot found: {self.param_path}"
            )

        with self.param_path.open("r", encoding="utf-8") as file:
            return yaml.safe_load(file)[
                "model_training"
            ]  # Only return the model_training section

    def _validate_algo_names(self) -> None:
        """
        Reject config keys that aren't safe to use as filenames/identifiers.

        Algo names become output filename stems (e.g. "RandomForest_seed_42")
        and dict keys / log identifiers throughout training. Catching a bad
        name here -- at config-load time -- gives a clear, immediate error
        instead of a confusing failure (or a silently wrong path) partway
        through a training run.
        """

        invalid_names = [
            name
            for name in self.hyperparams_config["models"]
            if not _VALID_ALGO_NAME_PATTERN.match(name)
        ]
        if invalid_names:
            raise ValueError(
                f"Invalid model name(s) in config: {invalid_names}. "
                "Model names must start with a letter and contain only "
                "letters, digits, underscores, and hyphens (e.g. "
                "'RandomForest', 'xgboost_v2')."
            )

    def _load_training_data(self, feature_path: Path | None = None) -> pd.DataFrame:
        """
        Load engineered feature data for model training.
        """

        feature_path: Path = feature_path or self.train_file_path

        if not feature_path.exists():
            raise FileNotFoundError(f"Feature file not found: {feature_path}")

        return pd.read_parquet(feature_path)

    @staticmethod
    def _split_features_and_target(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
        """
        Separate feature columns from the target label column.
        """

        return df.drop(columns=[TARGET_COLUMN]), df[TARGET_COLUMN]

    @staticmethod
    def _build_estimator(model_type: str, seed: int) -> Any:
        """
        Build an estimator of the specified type, passing the random seed if supported.
        """

        model_class = _resolve_model_class(model_type)
        constructor_params = inspect.signature(model_class.__init__).parameters
        if "random_state" in constructor_params:
            return model_class(random_state=seed)

        logger.debug(
            "%s has no random_state parameter; constructing without one.",
            model_type,
        )
        return model_class()

    @staticmethod
    def _save_trained_model(model: Any, model_path: Path | str) -> Path:
        """
        Save the trained model to disk, choosing the appropriate serialization format.
        """

        if not isinstance(model_path, Path):
            model_path = Path(model_path)

        create_dir(model_path.parent)

        if hasattr(model, "save_model"):
            resolved_path = model_path.with_suffix(".json")
            model.save_model(str(resolved_path))
        else:
            resolved_path = model_path.with_suffix(".joblib")
            joblib.dump(model, resolved_path)

        return resolved_path

    @staticmethod
    def _degrid_estimator_parallelism(params: dict[str, Any]) -> dict[str, Any]:
        """
        Override any hyperparameter that requests internal parallelism (e.g. n_jobs) to a single job, to avoid CPU oversubscription when GridSearchCV is also parallelizing across folds and candidates. This is a common source of performance degradation and confusing errors, so we enforce it here rather than relying on the user to remember to do it in config.
        """

        cleaned = dict(params)
        for key in _ESTIMATOR_PARALLELISM_PARAMS:
            if key in cleaned:
                logger.debug(
                    "Overriding grid param '%s'=%s to [1] to avoid CPU "
                    "oversubscription under GridSearchCV(n_jobs=-1)",
                    key,
                    cleaned[key],
                )
                cleaned[key] = [1]
        return cleaned

    def fetch_algo_names(self) -> list[str]:
        """
        Fetch the name of every model in the config with `enabled: true` (the default).
        """

        return [
            name
            for name, settings in self.hyperparams_config["models"].items()
            if settings.get("enabled", True)
        ]

    def train(self, algo_name: str = "XGBoost", feature_path: Path | None = None) -> Path:
        """
        Train one desired model and save it in the directory.
        """

        result, _ = self._fit_and_save_model(algo_name=algo_name, feature_path=feature_path)
        if result.model_path is None:
            raise RuntimeError(result.error or f"Training failed for {algo_name}")
        return result.model_path

    def train_one(
        self, algo_name: str = "XGBoost", feature_path: Path | None = None
        ) -> TrainingResult:
        """
        Train one model and return a structured, serializable result.
        """

        result, _ = self._fit_and_save_model(algo_name=algo_name, feature_path=feature_path)
        return result

    def _fit_and_save_model(
        self, algo_name: str = "XGBoost", feature_path: Path | None = None
        ) -> tuple[TrainingResult, Any]:
        """
        Train one model and return its result plus the fitted estimator for logging.
        """

        seed = self.params_config["seed"]
        df = self._load_training_data(feature_path=feature_path)
        features, target = self._split_features_and_target(df)

        if algo_name not in self.hyperparams_config["models"]:
            available = ", ".join(self.hyperparams_config["models"])
            raise ValueError(
                f"Unknown algo_name '{algo_name}'. Available models: {available}"
            )

        settings = self.hyperparams_config["models"][algo_name]
        estimator = self._build_estimator(settings["type"], seed)

        cv_splitter = StratifiedKFold(
            n_splits=self.params_config["cv_folds"],
            shuffle=False,
        )
        param_grid = self._degrid_estimator_parallelism(settings["params"])

        logger.info(
            "Running GridSearch for %s with %d-fold stratified CV",
            algo_name,
            self.params_config["cv_folds"],
        )
        grid_search = GridSearchCV(
            estimator,
            param_grid,
            scoring=self.params_config["scoring"],
            cv=cv_splitter,
            n_jobs=-1,
            verbose=2,
        )
        grid_search.fit(features, target)

        output_dir = create_dir(self.model_dir)
        model_path = output_dir / f"{algo_name}_seed_{seed}"
        model_path = self._save_trained_model(
            model=grid_search.best_estimator_, model_path=model_path
        )

        logger.info(
            "Saved model %s with best score %.4f to %s",
            model_path.name,
            grid_search.best_score_,
            output_dir,
        )

        result = TrainingResult(
            algo_name=algo_name,
            model_path=model_path,
            best_score=float(grid_search.best_score_),
            best_params=grid_search.best_params_,
        )
        return result, grid_search.best_estimator_

    def train_all(
        self,
        stop_on_error: bool = False,
        feature_path: Path | None = None,
        algo_names: list[str] | None = None,
        ) -> dict[str, str]:
        """
        Train every model in the config with `enabled: true` (the default).

        Models can be excluded by setting `enabled: false` in their config
        block. By default, one model's failure is caught and recorded so a
        bad grid on one algorithm doesn't stop the others from training;
        pass stop_on_error=True to raise immediately instead.

        Returns a dict mapping algo_name to saved model path on success, or the
        error message on failure.
        """

        algo_names = algo_names or self.fetch_algo_names()
        if not algo_names:
            logger.warning("No enabled models found in config; nothing to train.")

        results: dict[str, str] = {}

        pipeline_params = {
            "seed": self.seed,
            "cv_folds": self.params_config["cv_folds"],
            "scoring": self.params_config["scoring"],
        }

        tracker = MLflowTracker(self.param_path)

        with tracker.pipeline_run(seed=self.seed, params=pipeline_params) as run_context:
            if run_context is not None:
                tracker.log_artifact(self.hyperparams_config_path)

            for algo_name in algo_names:
                logger.info("=== Training '%s' ===", algo_name)
                with tracker.model_run(algo_name, run_context, stage="training") as run_id:
                    try:
                        result, fitted_model = self._fit_and_save_model(
                            algo_name=algo_name, feature_path=feature_path
                        )
                        results[algo_name] = result.to_model_paths_value()

                        tracker.log_training(
                            model=fitted_model,
                            algo_name=algo_name,
                            cv_best_score=float(result.best_score),
                            best_params=result.best_params,
                            scoring=self.params_config["scoring"],
                            seed=self.seed,
                            run_id=run_id,
                        )

                    except Exception as exc:
                        if stop_on_error:
                            raise
                        logger.error("Training failed for '%s': %s", algo_name, exc)
                        tracker.log_training_failure(str(exc))
                        results[algo_name] = str(exc)

        create_dir(SAVED_MODELS_PATH)
        ModelIO.save_json(
            data=results, path=SAVED_MODELS_PATH / f"model_paths_{self.seed}.json"
        )

        return results


def model_trainer(feature_path: Path, algo_name: str = "XGBoost") -> Path:
    """
    Train a single model and return the path of the saved model.
    """

    return ModelTrainer().train(algo_name=algo_name, feature_path=feature_path)


def train_all_models(
    feature_path: Path, stop_on_error: bool = False
) -> dict[str, str]:
    """
    Train all enabled models from the YAML file and return
    """

    return ModelTrainer().train_all(feature_path=feature_path, stop_on_error=stop_on_error)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    trainer = ModelTrainer()
    trainer.train_all(stop_on_error=False)
