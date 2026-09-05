"""
MLflow experiment tracking and model registry helpers.

Tracking URI resolution order:
  1. MLFLOW_TRACKING_URI environment variable
  2. configs/params.yaml mlflow.tracking_uri
  3. Local file store at {PROJECT_ROOT}/mlruns
"""

from __future__ import annotations

import json
import logging
import os
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator

import mlflow
import yaml
from mlflow.tracking import MlflowClient

from fraud_detection.paths import (
    MLFLOW_DIR,
    PARAMS_CONFIG_PATH,
    PROJECT_ROOT,
    create_dir,
    mlflow_run_context_path,
)

logger = logging.getLogger(__name__)


@dataclass
class MLflowConfig:
    """Resolved MLflow settings from params.yaml."""

    enabled: bool = True
    experiment_name: str = "fraud-detection"
    tracking_uri: str | None = None
    register_best_by: str = "min_cost"
    model_registry_name: str = "fraud-detection-model"


@dataclass
class RunContext:
    """Persisted parent-run metadata shared across DVC stages."""

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


def load_mlflow_config(params_path: Path = PARAMS_CONFIG_PATH) -> MLflowConfig:
    """Read MLflow settings from params.yaml."""

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
    """Resolve tracking URI: env var > config > local file store."""

    env_uri = os.getenv("MLFLOW_TRACKING_URI")
    if env_uri:
        return env_uri
    if config.tracking_uri:
        return config.tracking_uri
    return f"file:{(PROJECT_ROOT / 'mlruns').as_posix()}"


def setup_mlflow(params_path: Path = PARAMS_CONFIG_PATH) -> MLflowConfig:
    """Configure MLflow tracking URI and experiment."""

    config = load_mlflow_config(params_path)
    if not config.enabled:
        logger.info("MLflow tracking is disabled in config.")
        return config

    tracking_uri = resolve_tracking_uri(config)
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(config.experiment_name)
    logger.info(
        "MLflow configured: uri=%s experiment=%s",
        tracking_uri,
        config.experiment_name,
    )
    return config


def _save_run_context(context: RunContext) -> None:
    create_dir(MLFLOW_DIR)
    path = mlflow_run_context_path(context.seed)
    with path.open("w", encoding="utf-8") as file:
        json.dump(context.to_dict(), file, indent=2)


def load_run_context(seed: int | str) -> RunContext | None:
    """Load persisted parent-run context for a seed."""

    path = mlflow_run_context_path(seed)
    if not path.exists():
        return None

    with path.open("r", encoding="utf-8") as file:
        return RunContext.from_dict(json.load(file))


@contextmanager
def pipeline_run(
    seed: int | str,
    params: dict[str, Any] | None = None,
    tags: dict[str, str] | None = None,
    params_path: Path = PARAMS_CONFIG_PATH,
) -> Iterator[RunContext | None]:
    """Create a parent MLflow run for the duration of a pipeline stage."""

    config = setup_mlflow(params_path)
    if not config.enabled:
        yield None
        return

    run_tags = {"seed": str(seed), "stage": "pipeline"}
    if tags:
        run_tags.update(tags)

    with mlflow.start_run(run_name=f"pipeline_seed_{seed}") as run:
        if params:
            mlflow.log_params(
                {k: str(v) for k, v in params.items() if v is not None}
            )
        mlflow.set_tags(run_tags)

        experiment = mlflow.get_experiment_by_name(config.experiment_name)
        experiment_id = (
            experiment.experiment_id if experiment else run.info.experiment_id
        )

        context = RunContext(
            parent_run_id=run.info.run_id,
            experiment_id=experiment_id,
            seed=seed,
        )
        _save_run_context(context)
        logger.info("Started pipeline run %s for seed %s", run.info.run_id, seed)
        yield context


def start_pipeline_run(
    seed: int | str,
    params: dict[str, Any] | None = None,
    tags: dict[str, str] | None = None,
    params_path: Path = PARAMS_CONFIG_PATH,
) -> RunContext | None:
    """
    Create a parent MLflow run for downstream DVC stages that run separately.

    Unlike ``pipeline_run``, this starts and immediately finishes the parent run
    so nested runs in later stages can attach via ``parent_run_id``.
    """

    config = setup_mlflow(params_path)
    if not config.enabled:
        return None

    run_tags = {"seed": str(seed), "stage": "pipeline"}
    if tags:
        run_tags.update(tags)

    with mlflow.start_run(run_name=f"pipeline_seed_{seed}") as run:
        if params:
            mlflow.log_params(
                {k: str(v) for k, v in params.items() if v is not None}
            )
        mlflow.set_tags(run_tags)

        experiment = mlflow.get_experiment_by_name(config.experiment_name)
        experiment_id = (
            experiment.experiment_id if experiment else run.info.experiment_id
        )

        context = RunContext(
            parent_run_id=run.info.run_id,
            experiment_id=experiment_id,
            seed=seed,
        )
        _save_run_context(context)
        logger.info("Started pipeline run %s for seed %s", run.info.run_id, seed)
        return context


def resume_pipeline_run(seed: int | str) -> RunContext | None:
    """Load run context and ensure MLflow is configured."""

    config = setup_mlflow()
    if not config.enabled:
        return None

    context = load_run_context(seed)
    if context is None:
        logger.warning("No MLflow run context found for seed %s", seed)
    return context


def update_model_run_context(
    seed: int | str,
    algo_name: str,
    run_id: str,
    model_uri: str,
) -> None:
    """Record a nested model run in the persisted context."""

    context = load_run_context(seed)
    if context is None:
        return

    context.model_runs[algo_name] = {
        "run_id": run_id,
        "model_uri": model_uri,
    }
    _save_run_context(context)


@contextmanager
def start_model_run(
    algo_name: str,
    context: RunContext | None,
    stage: str,
) -> Iterator[str | None]:
    """Open a nested MLflow run for one algorithm."""

    if context is None:
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
    """Log training metrics, hyperparams, and model artifact."""

    if context is None or run_id is None:
        return

    mlflow.log_metric("cv_best_score", cv_best_score)
    mlflow.log_param("scoring", scoring)
    for param_name, param_value in best_params.items():
        mlflow.log_param(f"best_{param_name}", param_value)

    model_uri = _log_model_artifact(model, algo_name)
    if model_uri:
        update_model_run_context(seed, algo_name, run_id, model_uri)


def _log_model_artifact(model: Any, algo_name: str) -> str | None:
    """Log model using the appropriate MLflow flavor."""

    artifact_path = "model"
    try:
        if hasattr(model, "save_model"):
            import mlflow.xgboost

            mlflow.xgboost.log_model(model, artifact_path=artifact_path)
        else:
            import mlflow.sklearn

            mlflow.sklearn.log_model(model, artifact_path=artifact_path)
    except Exception as exc:
        logger.error("Failed to log model artifact for %s: %s", algo_name, exc)
        return None

    active_run = mlflow.active_run()
    if active_run is None:
        return None
    return f"runs:/{active_run.info.run_id}/{artifact_path}"


def log_training_failure(error_message: str) -> None:
    """Tag a failed training run with the error message."""

    if mlflow.active_run() is None:
        return
    mlflow.set_tag("training_error", error_message[:250])


def log_threshold_run(
    threshold_info: dict[str, float],
    threshold_json_path: Path,
) -> None:
    """Log threshold optimization metrics and artifact."""

    if mlflow.active_run() is None:
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


def log_evaluation_run(
    metrics: dict[str, float],
    pr_curve_path: Path,
    shap_path: Path,
) -> None:
    """Log evaluation metrics and report artifacts."""

    if mlflow.active_run() is None:
        return

    mlflow.log_metrics(metrics)
    if pr_curve_path.exists():
        mlflow.log_artifact(str(pr_curve_path))
    if shap_path.exists():
        mlflow.log_artifact(str(shap_path))


@dataclass
class ModelCandidate:
    """One model's scores used for champion selection."""

    algo_name: str
    min_cost: float
    average_precision: float
    model_uri: str
    best_threshold: float
    run_id: str


def select_champion(candidates: list[ModelCandidate]) -> ModelCandidate | None:
    """Pick the best model by min_cost, tiebreak on average_precision."""

    if not candidates:
        return None

    return min(
        candidates,
        key=lambda c: (c.min_cost, -c.average_precision),
    )


def register_champion_model(
    champion: ModelCandidate,
    registry_name: str,
    params_path: Path = PARAMS_CONFIG_PATH,
) -> str | None:
    """Register the champion model and set the 'champion' alias."""

    config = load_mlflow_config(params_path)
    if not config.enabled:
        return None

    setup_mlflow(params_path)
    client = MlflowClient()

    try:
        model_version = mlflow.register_model(
            champion.model_uri,
            registry_name,
        )
        version = model_version.version

        client.set_registered_model_alias(
            registry_name,
            "champion",
            version,
        )
        client.set_model_version_tag(
            registry_name,
            version,
            "best_threshold",
            str(champion.best_threshold),
        )
        client.set_model_version_tag(
            registry_name,
            version,
            "algo_name",
            champion.algo_name,
        )
        client.set_model_version_tag(
            registry_name,
            version,
            "min_cost",
            str(champion.min_cost),
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
    except Exception as exc:
        logger.error("Failed to register champion model: %s", exc)
        return None
