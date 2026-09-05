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
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
import yaml
from sklearn.model_selection import GridSearchCV, StratifiedKFold

from fraud_detection.feature_schema import TARGET_COLUMN
from fraud_detection.mlflow_tracking import (
    log_training_failure,
    log_training_run,
    pipeline_run,
    start_model_run,
)
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

    def _load_training_data(self, csv_path: Path | None = None) -> pd.DataFrame:
        """
        Load engineered feature data for model training.
        """

        csv_path: Path = csv_path or self.train_file_path

        if not csv_path.exists():
            raise FileNotFoundError(f"Feature file not found: {csv_path}")

        return pd.read_csv(csv_path)

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

        if not model_path.exists():
            create_dir(model_path)

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
        Override any hyperparameter that requests internal parallelism (e.g. n_jobs)
        to a single job, to avoid CPU oversubscription when GridSearchCV is also parallelizing across folds and candidates. This is a common source of performance degradation and confusing errors, so we enforce it here rather than relying on the user to remember to do it in config.
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

    def train(self, algo_name: str = "XGBoost", csv_path: Path | None = None) -> Path:
        """
        Train one desired model and save it in the directory.
        """

        model_path, _ = self._train_model(algo_name=algo_name, csv_path=csv_path)
        return model_path

    def _train_model(
        self, algo_name: str = "XGBoost", csv_path: Path | None = None
    ) -> tuple[Path, GridSearchCV]:
        """
        Train one model and return the saved path plus the fitted GridSearchCV.
        """

        # 1. Load the feature data and split into features and target.
        seed = self.params_config["seed"]
        df = self._load_training_data(csv_path=csv_path)
        features, target = self._split_features_and_target(df)

        # 2. Validate the requested algorithm is supported
        if algo_name not in self.hyperparams_config["models"]:
            available = ", ".join(self.hyperparams_config["models"])
            raise ValueError(
                f"Unknown algo_name '{algo_name}'. Available models: {available}"
            )

        # 3. Build the estimator with the specified random seed.
        settings = self.hyperparams_config["models"][
            algo_name
        ]  # output: {"type": "XGBClassifier", "params": {...}}
        model_type = settings[
            "type"
        ]  # output: "XGBClassifier" or "RandomForestClassifier"
        estimator = self._build_estimator(
            model_type, seed
        )  # output: RandomForestClassifier(random_state=seed) or XGBClassifier(random_state=seed)

        # 4. Perform a grid search over the hyperparameter in the yaml file
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
            verbose=3,
        )
        grid_search.fit(features, target)

        # 6. Save the best model to the `artifacts directory.
        output_dir = create_dir(self.model_dir)
        model_path = (
            output_dir / f"{algo_name}_seed_{seed}"
            or f"{algo_name}_{self.train_file_path.name}"
        )
        model_path = self._save_trained_model(
            model=grid_search.best_estimator_, model_path=model_path
        )

        # 7. Log the best score and the path where the model was saved.
        logger.info(
            "Saved model %s with best score %.4f to %s",
            model_path.name,
            grid_search.best_score_,
            output_dir,
        )

        return model_path, grid_search

    def train_all(
        self,
        stop_on_error: bool = False,
        csv_path: Path | None = None,
    ) -> dict[str, Path | str | Exception]:
        """
        Train every model in the config with `enabled: true` (the default).

        Models can be excluded by setting `enabled: false` in their config
        block. By default, one model's failure is caught and recorded so a
        bad grid on one algorithm doesn't stop the others from training;
        pass stop_on_error=True to raise immediately instead.

        Returns a dict mapping algo_name -> (X_test, y_test) on success, or
        algo_name -> the raised exception on failure.
        """

        algo_names = self.fetch_algo_names()
        if not algo_names:
            logger.warning("No enabled models found in config; nothing to train.")

        results: dict[str, Path | str | Exception] = {}

        pipeline_params = {
            "seed": self.seed,
            "cv_folds": self.params_config["cv_folds"],
            "scoring": self.params_config["scoring"],
        }

        with pipeline_run(seed=self.seed, params=pipeline_params) as run_context:
            if run_context is not None:
                import mlflow

                mlflow.log_artifact(str(self.hyperparams_config_path))

            for algo_name in algo_names:
                logger.info("=== Training '%s' ===", algo_name)
                with start_model_run(
                    algo_name, run_context, stage="training"
                ) as run_id:
                    try:
                        model_path, grid_search = self._train_model(
                            algo_name=algo_name, csv_path=csv_path
                        )
                        results[algo_name] = str(model_path)

                        log_training_run(
                            model=grid_search.best_estimator_,
                            algo_name=algo_name,
                            cv_best_score=float(grid_search.best_score_),
                            best_params=grid_search.best_params_,
                            scoring=self.params_config["scoring"],
                            seed=self.seed,
                            context=run_context,
                            run_id=run_id,
                        )

                    except Exception as exc:
                        if stop_on_error:
                            raise
                        logger.error("Training failed for '%s': %s", algo_name, exc)
                        log_training_failure(str(exc))
                        results[algo_name] = str(exc)

        # saved the paths in a json file, to be used later during threshold optimization or model evaluation phase to load models.
        create_dir(SAVED_MODELS_PATH)
        ModelIO.save_json(
            data=results, path=SAVED_MODELS_PATH / f"model_paths_{self.seed}.json"
        )

        return results


def model_trainer(csv_path: Path, algo_name: str = "xgboost") -> Path:
    """
    Train a single model and return the path of the saved model.
    """

    return ModelTrainer().train(algo_name=algo_name, csv_path=csv_path)


def train_all_models(
    csv_path: Path, stop_on_error: bool = False
) -> dict[str, Path | str | Exception]:
    """
    Train all enabled models from the YAML file and return
    """

    return ModelTrainer().train_all(csv_path=csv_path, stop_on_error=stop_on_error)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    trainer = ModelTrainer()
    trainer.train_all(stop_on_error=False)
