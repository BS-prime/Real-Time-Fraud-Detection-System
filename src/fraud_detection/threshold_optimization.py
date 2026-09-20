"""
Cost-sensitive threshold optimization.
"""

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from sklearn.metrics import confusion_matrix

from fraud_detection.feature_schema import TARGET_COLUMN
from fraud_detection.mlflow_tracking import MLflowTracker
from fraud_detection.model_io import (
    align_features_to_model,
    load_model,
    predict_fraud_probability,
    save_json,
)
from fraud_detection.paths import (
    MODEL_DIR,
    PARAMS_CONFIG_PATH,
    SAVED_MODELS_PATH,
    THRESHOLD_DIR,
    create_dir,
    test_feature_file_path,
    threshold_summary_path,
)

logger = logging.getLogger(__name__)


class ThresholdOptimizer:
    """
    Search for a decision threshold that minimizes business cost.
    """

    def __init__(
        self,
        model_dir: str | Path = MODEL_DIR,
        threshold_dir: str | Path = THRESHOLD_DIR,
    ) -> None:
        self.model_dir = Path(model_dir)
        self.threshold_dir = Path(threshold_dir)

    @staticmethod
    def business_cost(
        y_true: pd.Series | np.ndarray,
        y_pred: np.ndarray,
        cost_fp: float,
        cost_fn: float,
    ) -> float:
        """
        Calculate the business cost of a binary fraud decision rule.
        """

        _, false_positives, false_negatives, _ = confusion_matrix(
            y_true,
            y_pred,
            labels=[0, 1],
        ).ravel()
        return (cost_fp * false_positives) + (cost_fn * false_negatives)

    @staticmethod
    def find_best_threshold(
        y_true: pd.Series | np.ndarray,
        probabilities: np.ndarray,
        cost_fp: float,
        cost_fn: float,
    ) -> dict[str, float]:
        best_threshold = 0.5
        min_cost = float("inf")

        for threshold in np.linspace(0.01, 0.99, 99):
            predictions = (probabilities >= threshold).astype(int)
            cost = ThresholdOptimizer.business_cost(
                y_true, predictions, cost_fp=cost_fp, cost_fn=cost_fn
            )

            if cost < min_cost:
                min_cost = cost
                best_threshold = float(threshold)

        return {
            "best_threshold": best_threshold,
            "min_cost": float(min_cost),
            "cost_fp": float(cost_fp),
            "cost_fn": float(cost_fn),
        }

    def optimize(
        self,
        X_test: pd.DataFrame,
        y_test: pd.Series,
        model_name: str,
        cost_fp: float = 1.0,
        cost_fn: float = 10.0,
        save: bool = True,
    ) -> tuple[np.ndarray, np.ndarray, dict[str, float], Path]:
        """
        Search for the decision threshold that minimizes business cost on test data.
        """

        model_path = self.model_dir / model_name
        model = load_model(model_path)
        X_test_aligned = align_features_to_model(
            X_test, model, fallback_columns=list(X_test.columns)
        )

        probabilities = predict_fraud_probability(model, X_test_aligned)
        threshold_info = self.find_best_threshold(
            y_true=y_test,
            probabilities=probabilities,
            cost_fp=cost_fp,
            cost_fn=cost_fn,
        )

        output_dir = create_dir(self.threshold_dir)
        output_path = output_dir / f"optimal_threshold_{model_path.stem}.json"
        if save:
            save_json(threshold_info, output_path)

        logger.info(
            "Saved threshold %s with best value %.2f and min cost %.2f",
            output_path.name,
            threshold_info["best_threshold"],
            threshold_info["min_cost"],
        )

        best_threshold = threshold_info["best_threshold"]
        predictions = (probabilities >= best_threshold).astype(int)
        return probabilities, predictions, threshold_info, output_path


def threshold_optimizer(
    X_test: pd.DataFrame,
    y_test: pd.Series,
    model_name: str,
    cost_fp: float = 1.0,
    cost_fn: float = 10.0,
    save: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """
    API for optimal threshold.
    """

    probabilities, predictions, _, _ = ThresholdOptimizer().optimize(
        X_test,
        y_test,
        model_name=model_name,
        cost_fp=cost_fp,
        cost_fn=cost_fn,
        save=save,
    )
    return probabilities, predictions


def _load_threshold_params(
    params_path: Path = PARAMS_CONFIG_PATH,
) -> dict[str, float | int]:
    with params_path.open("r", encoding="utf-8") as file:
        params = yaml.safe_load(file)
    threshold_params = params["threshold_optimization"]
    return {
        "seed": threshold_params["seed"],
        "cost_fp": threshold_params["cost_fp"],
        "cost_fn": threshold_params["cost_fn"],
    }


def _load_successful_model_paths(seed: int | str) -> dict[str, str]:
    paths_file = SAVED_MODELS_PATH / f"model_paths_{seed}.json"
    if not paths_file.exists():
        raise FileNotFoundError(f"Model paths file not found: {paths_file}")

    with paths_file.open("r", encoding="utf-8") as file:
        model_paths: dict[str, str] = json.load(file)

    successful: dict[str, str] = {}
    for algo_name, path_or_error in model_paths.items():
        path = Path(path_or_error)
        if path.exists() and path.suffix in {".json", ".joblib"}:
            successful[algo_name] = path_or_error
        else:
            logger.warning(
                "Skipping %s: has not a valid model path (%s)", algo_name, path_or_error
            )
    return successful


def _load_test_data(seed: int | str) -> tuple[pd.DataFrame, pd.Series]:
    test_path = test_feature_file_path(seed)
    if not test_path.exists():
        raise FileNotFoundError(f"Test feature file not found: {test_path}")

    df = pd.read_parquet(test_path)
    X_test = df.drop(columns=[TARGET_COLUMN])
    y_test = df[TARGET_COLUMN]
    return X_test, y_test


def run_threshold_optimization() -> dict[str, dict[str, float]]:
    """
    Optimize thresholds for all successfully trained models.
    """

    params = _load_threshold_params()
    seed = params["seed"]
    cost_fp = float(params["cost_fp"])
    cost_fn = float(params["cost_fn"])

    tracker = MLflowTracker()
    run_context = tracker.resume_pipeline_run(seed)
    model_paths = _load_successful_model_paths(seed)
    X_test, y_test = _load_test_data(seed)

    optimizer = ThresholdOptimizer()
    summary: dict[str, dict[str, float]] = {}

    for algo_name, model_path_str in model_paths.items():
        model_name = Path(model_path_str).name
        logger.info("=== Optimizing threshold for '%s' ===", algo_name)

        with tracker.model_run(algo_name, run_context, stage="threshold"):
            _, _, threshold_info, output_path = optimizer.optimize(
                X_test=X_test,
                y_test=y_test,
                model_name=model_name,
                cost_fp=cost_fp,
                cost_fn=cost_fn,
                save=True,
            )
            tracker.log_threshold(threshold_info, output_path)
            summary[algo_name] = threshold_info

    summary_path = threshold_summary_path(seed)
    create_dir(summary_path.parent)
    save_json(summary, summary_path)
    logger.info("Saved threshold summary to %s", summary_path)

    return summary


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    run_threshold_optimization()
