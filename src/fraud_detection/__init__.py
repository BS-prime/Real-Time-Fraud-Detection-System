"""Convenient public imports for the fraud detection package."""

from importlib import import_module

_EXPORTS = {
    "generate_transactions_data": "fraud_detection.data_ingestion",
    "feature_engineer": "fraud_detection.feature_engineering",
    "model_trainer": "fraud_detection.model_training",
    "threshold_optimizer": "fraud_detection.threshold_optimization",
    "model_evaluator": "fraud_detection.model_evaluation",
    "generate_monitoring_report": "fraud_detection.drift_monitoring",
    "run_training_pipeline": "fraud_detection.training_pipeline",
}

__all__ = list(_EXPORTS)


def __getattr__(name: str):
    """Load public helpers only when they are requested."""
    if name not in _EXPORTS:
        raise AttributeError(f"module 'fraud_detection' has no attribute '{name}'")

    module = import_module(_EXPORTS[name])
    value = getattr(module, name)
    globals()[name] = value
    return value
