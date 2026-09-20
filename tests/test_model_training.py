"""Tests for model training result helpers."""

from pathlib import Path

from fraud_detection.model_training import TrainingResult


def test_training_result_serializes_success_as_model_path():
    result = TrainingResult(
        algo_name="XGBoost",
        model_path=Path("artifacts/models/XGBoost_seed_42.json"),
        best_score=0.91,
        best_params={"max_depth": 3},
    )

    assert result.succeeded is True
    assert result.to_model_paths_value() == str(
        Path("artifacts/models/XGBoost_seed_42.json")
    )


def test_training_result_serializes_failure_as_error():
    result = TrainingResult(
        algo_name="BrokenModel",
        model_path=None,
        best_score=None,
        best_params={},
        error="unsupported model",
    )

    assert result.succeeded is False
    assert result.to_model_paths_value() == "unsupported model"
