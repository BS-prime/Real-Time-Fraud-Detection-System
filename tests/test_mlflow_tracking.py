"""Tests for MLflow tracking helpers."""

from fraud_detection.mlflow_tracking import (
    MLflowConfig,
    MLflowTracker,
    ModelCandidate,
    RunContext,
    load_mlflow_config,
    resolve_tracking_uri,
    select_champion,
)


def test_resolve_tracking_uri_prefers_env(monkeypatch, tmp_path):
    config = MLflowConfig(tracking_uri=str(tmp_path / "config_uri"))
    monkeypatch.setenv("MLFLOW_TRACKING_URI", "http://remote:5000")
    assert resolve_tracking_uri(config) == "http://remote:5000"


def test_resolve_tracking_uri_falls_back_to_local(tmp_path):
    config = MLflowConfig(tracking_uri=None)
    uri = resolve_tracking_uri(config)
    assert uri.startswith("sqlite:///")
    assert uri.endswith("mlflow.db")


def test_select_champion_by_min_cost():
    candidates = [
        ModelCandidate(
            algo_name="RandomForest",
            min_cost=120.0,
            average_precision=0.85,
            model_uri="runs:/abc/model",
            best_threshold=0.4,
            run_id="abc",
        ),
        ModelCandidate(
            algo_name="XGBoost",
            min_cost=95.0,
            average_precision=0.80,
            model_uri="runs:/def/model",
            best_threshold=0.97,
            run_id="def",
        ),
        ModelCandidate(
            algo_name="LogisticRegression",
            min_cost=95.0,
            average_precision=0.75,
            model_uri="runs:/ghi/model",
            best_threshold=0.6,
            run_id="ghi",
        ),
    ]

    champion = select_champion(candidates)
    assert champion is not None
    assert champion.algo_name == "XGBoost"
    assert champion.min_cost == 95.0


def test_select_champion_tiebreaks_on_average_precision():
    candidates = [
        ModelCandidate(
            algo_name="ModelA",
            min_cost=50.0,
            average_precision=0.70,
            model_uri="runs:/a/model",
            best_threshold=0.5,
            run_id="a",
        ),
        ModelCandidate(
            algo_name="ModelB",
            min_cost=50.0,
            average_precision=0.90,
            model_uri="runs:/b/model",
            best_threshold=0.5,
            run_id="b",
        ),
    ]

    champion = select_champion(candidates)
    assert champion is not None
    assert champion.algo_name == "ModelB"


def test_select_champion_empty_list():
    assert select_champion([]) is None


def test_run_context_round_trip(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "fraud_detection.mlflow_tracking.MLFLOW_DIR",
        tmp_path,
    )
    monkeypatch.setattr(
        "fraud_detection.paths.MLFLOW_DIR",
        tmp_path,
    )

    context = RunContext(
        parent_run_id="parent123",
        experiment_id="exp456",
        seed=42,
        model_runs={
            "XGBoost": {
                "run_id": "child789",
                "model_uri": "runs:/child789/model",
            }
        },
    )

    tracker = MLflowTracker()
    tracker.save_run_context(context)
    loaded = tracker.load_run_context(42)

    assert loaded is not None
    assert loaded.parent_run_id == "parent123"
    assert loaded.model_runs["XGBoost"]["model_uri"] == "runs:/child789/model"


def test_tracker_resume_returns_none_when_disabled(tmp_path):
    params_file = tmp_path / "params.yaml"
    params_file.write_text("mlflow:\n  enabled: false\n", encoding="utf-8")

    tracker = MLflowTracker(params_file)

    assert tracker.resume_pipeline_run(seed=42) is None


def test_load_mlflow_config_defaults(tmp_path):
    params_file = tmp_path / "params.yaml"
    params_file.write_text("mlflow:\n  enabled: false\n", encoding="utf-8")

    config = load_mlflow_config(params_file)
    assert config.enabled is False
    assert config.experiment_name == "fraud-detection"
