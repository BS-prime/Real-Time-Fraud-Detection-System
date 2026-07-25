"""Model evaluation and report generation."""

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


def _shap_values_for_plot(model, features: pd.DataFrame):
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(features)

    if isinstance(shap_values, list):
        return shap_values[1]
    if getattr(shap_values, "ndim", 2) == 3:
        return shap_values[:, :, 1]
    return shap_values


def _plot_shap_summary(model, features: pd.DataFrame, output_path: Path) -> None:
    shap_values = _shap_values_for_plot(model, features)
    plt.figure()
    shap.summary_plot(shap_values, features, plot_type="bar", show=False)
    plt.savefig(output_path, bbox_inches="tight", dpi=300)
    plt.close()


def model_evaluator(
    model_name: str,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    y_prob: np.ndarray | pd.Series,
    y_pred_final: np.ndarray | pd.Series,
) -> dict[str, float | str]:
    """Evaluate a trained model and save precision-recall and SHAP reports."""
    model_path = MODEL_DIR / model_name
    model = load_model(model_path)
    X_test = align_features_to_model(X_test, model, fallback_columns=list(X_test.columns))

    model_stem = Path(model_name).stem
    evaluation_dir = ensure_directory(REPORTS_DIR / "model_evaluation")
    shap_dir = ensure_directory(REPORTS_DIR / "shap")

    pr_curve_path = evaluation_dir / f"precision_recall_curve_{model_stem}.png"
    shap_path = shap_dir / f"shap_summary_{model_stem}.png"

    average_precision = _plot_precision_recall_curve(y_test, y_prob, pr_curve_path)

    print(f"AUC-PR for {model_name}: {average_precision:.4f}\n")
    print(f"Classification Report for {model_name}:\n")
    print(classification_report(y_test, y_pred_final))

    _plot_shap_summary(model, X_test, shap_path)

    return {
        "average_precision": average_precision,
        "precision_recall_curve": str(pr_curve_path),
        "shap_summary": str(shap_path),
    }
