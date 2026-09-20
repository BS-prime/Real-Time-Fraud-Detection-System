"""
Model evaluation and report generation Module.
"""

import json
import logging
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
import yaml
from shap.utils._exceptions import InvalidModelError
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    classification_report,
    precision_recall_curve,
    precision_score,
    recall_score,
)

from fraud_detection.mlflow_tracking import (
    MLflowTracker,
    ModelCandidate,
    select_champion,
)
from fraud_detection.model_io import (
    align_features_to_model,
    load_model,
    predict_fraud_probability,
    save_json,
)
from fraud_detection.paths import (
    MODEL_DIR,
    PARAMS_CONFIG_PATH,
    REPORTS_DIR,
    create_dir,
    evaluation_summary_path,
    threshold_summary_path,
)
from fraud_detection.threshold_optimization import (
    _load_successful_model_paths,
    _load_test_data,
)

logger = logging.getLogger(__name__)


class ModelEvaluator:
    """
    Evaluate a trained model and produce plots and summary metrics.
    """

    def __init__(
        self,
        model_dir: Path = MODEL_DIR,
        reports_dir: Path = REPORTS_DIR,
        params_config_path: Path = PARAMS_CONFIG_PATH,
    ) -> None:
        self.model_dir = model_dir
        self.reports_dir = reports_dir
        self.params_path = params_config_path
        self.params = self._load_params()
        self.seed = self.params["seed"]

    def _load_params(self) -> dict[str, Any]:
        """
        Read the model evaluation params from YAML config.
        """

        if not self.params_path.exists():
            raise FileNotFoundError(
                f"Params config file not found!!!: {self.params_path}"
            )

        with self.params_path.open("r", encoding="utf-8") as file:
            return yaml.safe_load(file)["model_evaluation"]

    @staticmethod
    def _plot_precision_recall_curve(
        y_test: pd.Series,
        y_prob: np.ndarray | pd.Series,
        output_path: Path,
    ) -> float:
        """
        plot the curve and save as a png file.
        """

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
    def _shap_explainer(model, features: pd.DataFrame):
        """
        Pick a SHAP explainer that matches the estimator family.
        """

        try:
            return shap.TreeExplainer(model)
        except InvalidModelError:
            return shap.LinearExplainer(model, features)

    @staticmethod
    def _positive_class_shap_values(shap_values):
        """
        Reduce multi-output SHAP arrays to the fraud (positive) class.
        """

        if hasattr(shap_values, "values"):
            shap_values = shap_values.values
        if isinstance(shap_values, list):
            return shap_values[1]
        if getattr(shap_values, "ndim", 2) == 3:
            return shap_values[:, :, 1]
        return shap_values

    @classmethod
    def _shap_values_for_plot(cls, model, features: pd.DataFrame):
        """
        Calculate how each feature contributes to the prediction.
        """

        explainer = cls._shap_explainer(model, features)
        return cls._positive_class_shap_values(explainer.shap_values(features))

    def _plot_shap_summary(
        self, model, features: pd.DataFrame, output_path: Path
    ) -> None:
        """
        save the shap feature contribution plot as png file.
        """

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
        """
        Evaluate a trained model and generate evaluation plots and metrics.
        """

        model_path = self.model_dir / model_name
        model = load_model(model_path)
        X_test_aligned = align_features_to_model(
            X_test, model, fallback_columns=list(X_test.columns)
        )

        model_stem = Path(model_name).stem
        evaluation_dir = create_dir(self.reports_dir / "model_evaluation")
        shap_dir = create_dir(self.reports_dir / "shap")

        pr_curve_path = evaluation_dir / f"precision_recall_curve_{model_stem}.png"
        shap_path = shap_dir / f"shap_summary_{model_stem}.png"

        average_precision = self._plot_precision_recall_curve(
            y_test, y_prob, pr_curve_path
        )
        accuracy = accuracy_score(y_test, y_pred_final)
        precision = precision_score(y_test, y_pred_final)
        recall = recall_score(y_test, y_pred_final)

        logger.info("AUC-PR for %s: %.4f", model_name, average_precision)
        logger.info(
            "Classification report for %s:\n%s",
            model_name,
            classification_report(y_test, y_pred_final),
        )

        self._plot_shap_summary(model, X_test_aligned, shap_path)

        result: dict[str, dict[str, float]] = {
            model_name: {
                "accuracy": accuracy,
                "precision": precision,
                "recall": recall,
                "average_precision": average_precision,
            }
        }

        save_json(result, path=evaluation_dir / f"{model_name}.json")

        return {
            "average_precision": average_precision,
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
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
    """
    Evaluate a trained model and generate evaluation plots and metrics.
    """

    return ModelEvaluator().evaluate(model_name, X_test, y_test, y_prob, y_pred_final)


def _load_threshold_summary(seed: int | str) -> dict[str, dict[str, float]]:
    summary_path = threshold_summary_path(seed)
    if not summary_path.exists():
        raise FileNotFoundError(f"Threshold summary not found: {summary_path}")

    with summary_path.open("r", encoding="utf-8") as file:
        return json.load(file)


def _predict_at_threshold(
    model_name: str,
    X_test: pd.DataFrame,
    best_threshold: float,
    model_dir: Path = MODEL_DIR,
) -> tuple[np.ndarray, np.ndarray]:
    model_path = model_dir / model_name
    model = load_model(model_path)
    X_test_aligned = align_features_to_model(
        X_test, model, fallback_columns=list(X_test.columns)
    )
    probabilities = predict_fraud_probability(model, X_test_aligned)
    return probabilities, (probabilities >= best_threshold).astype(int)


def run_model_evaluation() -> dict[str, dict[str, float | str]]:
    """
    Evaluate all successfully trained models and register the champion.
    """

    evaluator = ModelEvaluator()
    seed = evaluator.seed

    tracker = MLflowTracker()
    run_context = tracker.resume_pipeline_run(seed)
    model_paths = _load_successful_model_paths(seed)
    threshold_summary = _load_threshold_summary(seed)
    X_test, y_test = _load_test_data(seed)

    evaluation_summary: dict[str, dict[str, float | str]] = {}
    candidates: list[ModelCandidate] = []

    for algo_name, model_path_str in model_paths.items():
        if algo_name not in threshold_summary:
            logger.warning("No threshold info for %s; skipping evaluation", algo_name)
            continue

        model_name = Path(model_path_str).name
        threshold_info = threshold_summary[algo_name]
        best_threshold = threshold_info["best_threshold"]

        logger.info("=== Evaluating '%s' ===", algo_name)

        with tracker.model_run(algo_name, run_context, stage="evaluation"):
            y_prob, y_pred = _predict_at_threshold(model_name, X_test, best_threshold)
            metrics = evaluator.evaluate(
                model_name=model_name,
                X_test=X_test,
                y_test=y_test,
                y_prob=y_prob,
                y_pred_final=y_pred,
            )

            tracker.log_evaluation(
                metrics={
                    "accuracy": float(metrics["accuracy"]),
                    "precision": float(metrics["precision"]),
                    "recall": float(metrics["recall"]),
                    "average_precision": float(metrics["average_precision"]),
                },
                pr_curve_path=Path(str(metrics["precision_recall_curve"])),
                shap_path=Path(str(metrics["shap_summary"])),
            )

            evaluation_summary[algo_name] = {
                **{
                    k: float(v)
                    for k, v in metrics.items()
                    if isinstance(v, (int, float))
                },
                "best_threshold": best_threshold,
                "min_cost": threshold_info["min_cost"],
            }

            mlflow_context = tracker.load_run_context(seed)
            if mlflow_context and algo_name in mlflow_context.model_runs:
                model_run = mlflow_context.model_runs[algo_name]
                candidates.append(
                    ModelCandidate(
                        algo_name=algo_name,
                        min_cost=float(threshold_info["min_cost"]),
                        average_precision=float(metrics["average_precision"]),
                        model_uri=model_run["model_uri"],
                        best_threshold=float(best_threshold),
                        run_id=model_run["run_id"],
                    )
                )

    summary_path = evaluation_summary_path(seed)
    create_dir(summary_path.parent)
    save_json(evaluation_summary, summary_path)
    logger.info("Saved evaluation summary to %s", summary_path)

    champion = select_champion(candidates)
    if champion is not None:
        tracker.register_champion(champion)
        evaluation_summary["_champion"] = {
            "algo_name": champion.algo_name,
            "min_cost": champion.min_cost,
            "average_precision": champion.average_precision,
            "best_threshold": champion.best_threshold,
        }
        save_json(evaluation_summary, summary_path)

    return evaluation_summary


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    run_model_evaluation()
