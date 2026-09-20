"""
MLflow experiment tracking and model registry helpers.

Tracking URI resolution order:
  1. MLFLOW_TRACKING_URI environment variable
  2. configs/params.yaml mlflow.tracking_uri
  3. Local SQLite store at {PROJECT_ROOT}/mlflow.db
"""

from __future__ import annotations

import json
import logging
import os
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import mlflow
import yaml
from mlflow.exceptions import MlflowException
from mlflow.tracking import MlflowClient

from fraud_detection.paths import (
    MLFLOW_DIR,
    MLFLOW_TRACKING_DB,
    PARAMS_CONFIG_PATH,
    create_dir,
    mlflow_run_context_path,
)

logger = logging.getLogger(__name__)


@dataclass
class MLflowConfig:
    """
    MLflow configuration.
    """

    enabled: bool = True
    experiment_name: str = "fraud-detection"
    tracking_uri: str | None = None
    register_best_by: str = "min_cost"
    model_registry_name: str = "fraud-detection-model"


@dataclass
class RunContext:
    """
    Persisted parent-run metadata shared across DVC stages.
    """

    parent_run_id: str
    experiment_id: str
    seed: int | str
    model_runs: dict[str, dict[str, str]] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "parent_run_id": self.parent_run_id,
            "experiment_id": self.experiment_id,
            "seed": self.seed,
            "model_runs": self.model_runs,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> RunContext:
        return cls(
            parent_run_id=data["parent_run_id"],
            experiment_id=data["experiment_id"],
            seed=data["seed"],
            model_runs=data.get("model_runs", {}),
        )


@dataclass
class ModelCandidate:
    """
    One model's scores used for champion selection.
    """

    algo_name: str
    min_cost: float
    average_precision: float
    model_uri: str
    best_threshold: float
    run_id: str


def load_mlflow_config(params_path: Path = PARAMS_CONFIG_PATH) -> MLflowConfig:
    """
    Read MLflow settings from params.yaml.
    """

    if not params_path.exists():
        return MLflowConfig()

    with params_path.open("r", encoding="utf-8") as file:
        params = yaml.safe_load(file) or {}

    mlflow_params = params.get("mlflow", {})
    return MLflowConfig(
        enabled=mlflow_params.get("enabled", True),
        experiment_name=mlflow_params.get("experiment_name", "fraud-detection"),
        tracking_uri=mlflow_params.get("tracking_uri"),
        register_best_by=mlflow_params.get("register_best_by", "min_cost"),
        model_registry_name=mlflow_params.get(
            "model_registry_name", "fraud-detection-model"
        ),
    )


def resolve_tracking_uri(config: MLflowConfig) -> str:
    """
    Resolve tracking URI: env var > config > local SQLite store.
    """

    env_uri = os.getenv("MLFLOW_TRACKING_URI")
    if env_uri:
        return env_uri
    if config.tracking_uri:
        return config.tracking_uri
    return f"sqlite:///{MLFLOW_TRACKING_DB.as_posix()}"


class MLflowTracker:
    """
    MLflow tracker class.
    """

    def __init__(self, params_path: Path = PARAMS_CONFIG_PATH) -> None:
        self.params_path = params_path
        self.config = load_mlflow_config(params_path)

    @property
    def enabled(self) -> bool:
        """
        Check if MLflow tracking is enabled.
        """

        return self.config.enabled

    def setup(self) -> MLflowConfig:
        """
        Setup MLflow tracking.
        """

        if not self.config.enabled:
            logger.info("MLflow tracking is disabled in config.")
            return self.config

        tracking_uri = resolve_tracking_uri(self.config)
        mlflow.set_tracking_uri(tracking_uri)
        mlflow.set_experiment(self.config.experiment_name)
        logger.info(
            "MLflow configured: uri=%s experiment=%s",
            tracking_uri,
            self.config.experiment_name,
        )
        return self.config

    def start_pipeline_run(
        self,
        seed: int | str,
        params: dict[str, Any] | None = None,
        tags: dict[str, str] | None = None,
        ) -> RunContext | None:
        """
        Create and finish a parent MLflow run for downstream DVC stages.
        """

        self.setup()
        if not self.enabled:
            return None

        run_tags = {"seed": str(seed), "stage": "pipeline"}
        if tags:
            run_tags.update(tags)

        with mlflow.start_run(run_name=f"pipeline_seed_{seed}") as run:
            if params:
                mlflow.log_params(
                    {
                        key: str(value)
                        for key, value in params.items()
                        if value is not None
                    }
                )
            mlflow.set_tags(run_tags)

            experiment = mlflow.get_experiment_by_name(self.config.experiment_name)
            experiment_id = (
                experiment.experiment_id if experiment else run.info.experiment_id
            )
            context = RunContext(
                parent_run_id=run.info.run_id,
                experiment_id=experiment_id,
                seed=seed,
            )
            self.save_run_context(context)
            logger.info("Started pipeline run %s for seed %s", run.info.run_id, seed)
            return context

    @contextmanager
    def pipeline_run(
        self,
        seed: int | str,
        params: dict[str, Any] | None = None,
        tags: dict[str, str] | None = None,
        ) -> Iterator[RunContext | None]:
        """
        Create a parent MLflow run and keep it active for nested runs in one process.
        """

        self.setup()
        if not self.enabled:
            yield None
            return

        run_tags = {"seed": str(seed), "stage": "pipeline"}
        if tags:
            run_tags.update(tags)

        with mlflow.start_run(run_name=f"pipeline_seed_{seed}") as run:
            if params:
                mlflow.log_params(
                    {
                        key: str(value)
                        for key, value in params.items()
                        if value is not None
                    }
                )
            mlflow.set_tags(run_tags)

            experiment = mlflow.get_experiment_by_name(self.config.experiment_name)
            experiment_id = (
                experiment.experiment_id if experiment else run.info.experiment_id
            )
            context = RunContext(
                parent_run_id=run.info.run_id,
                experiment_id=experiment_id,
                seed=seed,
            )
            self.save_run_context(context)
            logger.info("Started pipeline run %s for seed %s", run.info.run_id, seed)
            yield context

    def resume_pipeline_run(self, seed: int | str) -> RunContext | None:
        """
        Load run context and ensure MLflow is configured.
        """

        self.setup()
        if not self.enabled:
            return None

        context = self.load_run_context(seed)
        if context is None:
            logger.warning("No MLflow run context found for seed %s", seed)
        return context

    @contextmanager
    def model_run(
        self,
        algo_name: str,
        context: RunContext | None,
        stage: str,
        ) -> Iterator[str | None]:
        """
        Open a nested MLflow run for one algorithm.
        """

        if context is None or not self.enabled:
            yield None
            return

        with mlflow.start_run(
            run_name=f"{algo_name}_{stage}",
            nested=True,
            parent_run_id=context.parent_run_id,
        ) as run:
            mlflow.set_tags(
                {
                    "algo_name": algo_name,
                    "seed": str(context.seed),
                    "stage": stage,
                }
            )
            yield run.info.run_id

    def log_artifact(self, artifact_path: Path) -> None:
        """
        Log an artifact to MLflow.
        """

        if self.enabled and mlflow.active_run() is not None and artifact_path.exists():
            mlflow.log_artifact(str(artifact_path))

    def log_training(
        self,
        model: Any,
        algo_name: str,
        cv_best_score: float,
        best_params: dict[str, Any],
        scoring: str,
        seed: int | str,
        run_id: str | None,
        ) -> None:
        """
        Log training metrics, hyperparams, and model artifact.
        """

        if run_id is None or not self.enabled:
            return

        mlflow.log_metric("cv_best_score", cv_best_score)
        mlflow.log_param("scoring", scoring)
        for param_name, param_value in best_params.items():
            mlflow.log_param(f"best_{param_name}", param_value)

        model_uri = self._log_model_artifact(model, algo_name)
        if model_uri:
            self.update_model_run_context(seed, algo_name, run_id, model_uri)

    def log_training_failure(self, error_message: str) -> None:
        """
        Tag a failed training run with the error message.
        """

        if self.enabled and mlflow.active_run() is not None:
            mlflow.set_tag("training_error", error_message[:250])

    def log_threshold(
        self,
        threshold_info: dict[str, float],
        threshold_json_path: Path,
        ) -> None:
        """
        Log threshold optimization metrics and artifact.
        """

        if not self.enabled or mlflow.active_run() is None:
            return

        mlflow.log_metrics(
            {
                "best_threshold": threshold_info["best_threshold"],
                "min_cost": threshold_info["min_cost"],
            }
        )
        mlflow.log_params(
            {
                "cost_fp": threshold_info["cost_fp"],
                "cost_fn": threshold_info["cost_fn"],
            }
        )
        if threshold_json_path.exists():
            mlflow.log_artifact(str(threshold_json_path))

    def log_evaluation(
        self,
        metrics: dict[str, float],
        pr_curve_path: Path,
        shap_path: Path,
        ) -> None:
        """
        Log evaluation metrics and report artifacts.
        """

        if not self.enabled or mlflow.active_run() is None:
            return

        mlflow.log_metrics(metrics)
        if pr_curve_path.exists():
            mlflow.log_artifact(str(pr_curve_path))
        if shap_path.exists():
            mlflow.log_artifact(str(shap_path))

    def register_champion(
        self,
        champion: ModelCandidate,
        registry_name: str | None = None,
        ) -> str | None:
        """
        Register the champion model and set the 'champion' alias.
        """

        self.setup()
        if not self.enabled:
            return None

        registry_name = registry_name or self.config.model_registry_name
        client = MlflowClient()

        try:
            model_version = mlflow.register_model(champion.model_uri, registry_name)
            version = model_version.version

            client.set_registered_model_alias(registry_name, "champion", version)
            client.set_model_version_tag(
                registry_name, version, "best_threshold", str(champion.best_threshold)
            )
            client.set_model_version_tag(
                registry_name, version, "algo_name", champion.algo_name
            )
            client.set_model_version_tag(
                registry_name, version, "min_cost", str(champion.min_cost)
            )
            client.set_model_version_tag(
                registry_name,
                version,
                "average_precision",
                str(champion.average_precision),
            )

            logger.info(
                "Registered champion %s as %s version %s (alias: champion)",
                champion.algo_name,
                registry_name,
                version,
            )
            return version
        except MlflowException as exc:
            logger.error("Failed to register champion model: %s", exc)
            return None

    def save_run_context(self, context: RunContext) -> None:
        """
        Save the run context to a file.
        """

        create_dir(MLFLOW_DIR)
        path = mlflow_run_context_path(context.seed)
        with path.open("w", encoding="utf-8") as file:
            json.dump(context.to_dict(), file, indent=2)

    def load_run_context(self, seed: int | str) -> RunContext | None:
        """
        Load the run context from a file.
        """

        path = mlflow_run_context_path(seed)
        if not path.exists():
            return None

        with path.open("r", encoding="utf-8") as file:
            return RunContext.from_dict(json.load(file))

    def update_model_run_context(   
        self,
        seed: int | str,
        algo_name: str,
        run_id: str,
        model_uri: str,
        ) -> None:
        """
        Update the model run context with the run ID and model URI.
        """

        context = self.load_run_context(seed)
        if context is None:
            return

        context.model_runs[algo_name] = {
            "run_id": run_id,
            "model_uri": model_uri,
        }
        self.save_run_context(context)

    @staticmethod
    def _log_model_artifact(model: Any, algo_name: str) -> str | None:
        """
        Log the model artifact to MLflow.
        """

        artifact_path = "model"
        try:
            if hasattr(model, "save_model"):
                import mlflow.xgboost

                mlflow.xgboost.log_model(model, artifact_path=artifact_path)
            else:
                import mlflow.sklearn

                mlflow.sklearn.log_model(model, artifact_path=artifact_path)
        except (MlflowException, OSError, TypeError, ValueError) as exc:
            logger.error("Failed to log model artifact for %s: %s", algo_name, exc)
            return None

        active_run = mlflow.active_run()
        if active_run is None:
            return None
        return f"runs:/{active_run.info.run_id}/{artifact_path}"


# MLflow tracker API
def setup_mlflow(params_path: Path = PARAMS_CONFIG_PATH) -> MLflowConfig:
    """
    Setup MLflow tracking.
    """

    return MLflowTracker(params_path).setup()


def load_run_context(seed: int | str) -> RunContext | None:
    """
    Load the run context from a file.
    """

    return MLflowTracker().load_run_context(seed)


@contextmanager
def pipeline_run(
    seed: int | str,
    params: dict[str, Any] | None = None,
    tags: dict[str, str] | None = None,
    params_path: Path = PARAMS_CONFIG_PATH,
    ) -> Iterator[RunContext | None]:
    """
    Create a parent MLflow run and keep it active for nested runs in one process.
    """

    with MLflowTracker(params_path).pipeline_run(seed, params=params, tags=tags) as ctx:
        yield ctx


def start_pipeline_run(
    seed: int | str,
    params: dict[str, Any] | None = None,
    tags: dict[str, str] | None = None,
    params_path: Path = PARAMS_CONFIG_PATH,
    ) -> RunContext | None:
    """
    Create and finish a parent MLflow run for downstream DVC stages.
    """

    return MLflowTracker(params_path).start_pipeline_run(seed, params=params, tags=tags)


def resume_pipeline_run(seed: int | str) -> RunContext | None:
    """
    Load run context and ensure MLflow is configured.
    """

    return MLflowTracker().resume_pipeline_run(seed)


@contextmanager
def start_model_run(
    algo_name: str,
    context: RunContext | None,
    stage: str,
    ) -> Iterator[str | None]:
    """
    Open a nested MLflow run for one algorithm.
    """

    with MLflowTracker().model_run(algo_name, context, stage) as run_id:
        yield run_id


def update_model_run_context(
    seed: int | str,
    algo_name: str,
    run_id: str,
    model_uri: str,
    ) -> None:
    """
    Update the model run context with the run ID and model URI.
    """

    MLflowTracker().update_model_run_context(seed, algo_name, run_id, model_uri)


def log_training_run(
    model: Any,
    algo_name: str,
    cv_best_score: float,
    best_params: dict[str, Any],
    scoring: str,
    seed: int | str,
    context: RunContext | None,
    run_id: str | None,
    ) -> None:
    """
    Log the training metrics, hyperparams, and model artifact.
    """

    if context is None:
        return
    MLflowTracker().log_training(
        model=model,
        algo_name=algo_name,
        cv_best_score=cv_best_score,
        best_params=best_params,
        scoring=scoring,
        seed=seed,
        run_id=run_id,
    )


def log_training_failure(error_message: str) -> None:
    """
    Tag a failed training run with the error message.
    """

    MLflowTracker().log_training_failure(error_message)


def log_threshold_run(
    threshold_info: dict[str, float],
    threshold_json_path: Path,
    ) -> None:
    """
    Log the threshold optimization metrics and artifact.
    """

    MLflowTracker().log_threshold(threshold_info, threshold_json_path)


def log_evaluation_run(
    metrics: dict[str, float],
    pr_curve_path: Path,
    shap_path: Path,
    ) -> None:
    """
    Log the evaluation metrics and report artifacts.
    """

    MLflowTracker().log_evaluation(metrics, pr_curve_path, shap_path)


def select_champion(candidates: list[ModelCandidate]) -> ModelCandidate | None:
    """
    Pick the best model by min_cost, tiebreak on average_precision.
    """

    if not candidates:
        return None

    return min(
        candidates,
        key=lambda candidate: (candidate.min_cost, -candidate.average_precision),
    )


def register_champion_model(
    champion: ModelCandidate,
    registry_name: str,
    params_path: Path = PARAMS_CONFIG_PATH,
    ) -> str | None:
    """
    Register the champion model and set the 'champion' alias.
    """

    return MLflowTracker(params_path).register_champion(champion, registry_name)
