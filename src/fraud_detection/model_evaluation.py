"""Model evaluation and report generation."""

import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from sklearn.metrics import (
    average_precision_score,
    classification_report,
    precision_recall_curve,
)

from fraud_detection.model_io import align_features_to_model, load_model
from fraud_detection.paths import MODEL_DIR, REPORTS_DIR, ensure_directory

logger = logging.getLogger(__name__)


class ModelEvaluator:
    """Evaluate a trained model and produce plots and summary metrics."""

    def __init__(
        self, model_dir: Path = MODEL_DIR, reports_dir: Path = REPORTS_DIR
    ) -> None:
        self.model_dir = model_dir
        self.reports_dir = reports_dir

    @staticmethod
    def _plot_precision_recall_curve(
        y_test: pd.Series,
        y_prob: np.ndarray | pd.Series,
        output_path: Path,
    ) -> float:
        average_precision = average_precision_score(y_test, y_prob)
        precision, recall, _ = precision_recall_curve(y_test, y_prob)

        plt.figure(figsize=(8, 6))
        plt.step(recall, precision, where="post")
        plt.xlabel("Recall")
        plt.ylabel("Precision")
        plt.title(f"Precision-Recall curve: AP={average_precision:.2f}")
        plt.savefig(output_path, bbox_inches="tight", dpi=300)
        plt.close()

        return float(average_precision)

    @staticmethod
    def _shap_values_for_plot(model, features: pd.DataFrame):
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(features)

        if isinstance(shap_values, list):
            return shap_values[1]
        if getattr(shap_values, "ndim", 2) == 3:
            return shap_values[:, :, 1]
        return shap_values

    def _plot_shap_summary(
        self, model, features: pd.DataFrame, output_path: Path
    ) -> None:
        shap_values = self._shap_values_for_plot(model, features)
        plt.figure()
        shap.summary_plot(shap_values, features, plot_type="bar", show=False)
        plt.savefig(output_path, bbox_inches="tight", dpi=300)
        plt.close()

    def evaluate(
        self,
        model_name: str,
        X_test: pd.DataFrame,
        y_test: pd.Series,
        y_prob: np.ndarray | pd.Series,
        y_pred_final: np.ndarray | pd.Series,
    ) -> dict[str, float | str]:
        model_path = self.model_dir / model_name
        model = load_model(model_path)
        X_test_aligned = align_features_to_model(
            X_test, model, fallback_columns=list(X_test.columns)
        )

        model_stem = Path(model_name).stem
        evaluation_dir = ensure_directory(self.reports_dir / "model_evaluation")
        shap_dir = ensure_directory(self.reports_dir / "shap")

        pr_curve_path = evaluation_dir / f"precision_recall_curve_{model_stem}.png"
        shap_path = shap_dir / f"shap_summary_{model_stem}.png"

        average_precision = self._plot_precision_recall_curve(
            y_test, y_prob, pr_curve_path
        )

        logger.info("AUC-PR for %s: %.4f", model_name, average_precision)
        logger.info(
            "Classification report for %s:\n%s",
            model_name,
            classification_report(y_test, y_pred_final),
        )

        self._plot_shap_summary(model, X_test_aligned, shap_path)

        return {
            "average_precision": average_precision,
            "precision_recall_curve": str(pr_curve_path),
            "shap_summary": str(shap_path),
        }


def model_evaluator(
    model_name: str,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    y_prob: np.ndarray | pd.Series,
    y_pred_final: np.ndarray | pd.Series,
) -> dict[str, float | str]:
    return ModelEvaluator().evaluate(model_name, X_test, y_test, y_prob, y_pred_final)
