"""
Cost-sensitive threshold optimization.
"""

from pathlib import Path
import logging
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix

from fraud_detection.model_io import (
    align_features_to_model,
    load_model,
    predict_fraud_probability,
    save_json,
)
from fraud_detection.paths import MODEL_DIR, THRESHOLD_DIR, ensure_directory

logger = logging.getLogger(__name__)


def business_cost(
    y_true: pd.Series | np.ndarray,
    y_pred: np.ndarray,
    cost_fp: float,
    cost_fn: float,
) -> float:
    """Calculate the business cost of a binary fraud decision rule."""
    _, false_positives, false_negatives, _ = confusion_matrix(
        y_true,
        y_pred,
        labels=[0, 1],
    ).ravel()
    return (cost_fp * false_positives) + (cost_fn * false_negatives)


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
            cost = business_cost(y_true, predictions, cost_fp=cost_fp, cost_fn=cost_fn)

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
    ) -> tuple[np.ndarray, np.ndarray]:
        """Search for the decision threshold that minimizes business cost on test data."""
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

        output_dir = ensure_directory(self.threshold_dir)
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
        return probabilities, predictions


def threshold_optimizer(
    X_test: pd.DataFrame,
    y_test: pd.Series,
    model_name: str,
    cost_fp: float = 1.0,
    cost_fn: float = 10.0,
    save: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    return ThresholdOptimizer().optimize(
        X_test,
        y_test,
        model_name=model_name,
        cost_fp=cost_fp,
        cost_fn=cost_fn,
        save=save,
    )
