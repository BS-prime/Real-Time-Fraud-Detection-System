"""Cost-sensitive threshold optimization."""

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


def business_cost(
    y_true: pd.Series | np.ndarray,
    y_pred: np.ndarray,
    cost_fp: float,
    cost_fn: float,
) -> float:
    """
    Calculate the business cost from false positives and false negatives.
    """
    _, false_positives, false_negatives, _ = confusion_matrix(
        y_true,
        y_pred,
        labels=[0, 1],
    ).ravel()
    return (cost_fp * false_positives) + (cost_fn * false_negatives)


def find_best_threshold(
    y_true: pd.Series | np.ndarray,
    probabilities: np.ndarray,
    cost_fp: float,
    cost_fn: float,
) -> dict[str, float]:
    """
    Search thresholds from 0.01 to 0.99 and keep the lowest-cost option.
    """
    
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


def threshold_optimizer(
    X_test: pd.DataFrame,
    y_test: pd.Series,
    model_name: str,
    cost_fp: float = 1.0,
    cost_fn: float = 10.0,
    save: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Find the model threshold that minimizes business cost.
    """

    model_path = MODEL_DIR / model_name
    model = load_model(model_path)
    X_test = align_features_to_model(X_test, model, fallback_columns=list(X_test.columns))

    probabilities = predict_fraud_probability(model, X_test)
    threshold_info = find_best_threshold(
        y_true=y_test,
        probabilities=probabilities,
        cost_fp=cost_fp,
        cost_fn=cost_fn,
    )

    output_dir = ensure_directory(THRESHOLD_DIR)
    output_path = output_dir / f"optimal_threshold_{model_path.stem}.json"
    if save:
        save_json(threshold_info, output_path)

    best_threshold = threshold_info["best_threshold"]
    predictions = (probabilities >= best_threshold).astype(int)

    print("=" * 70)
    print(f"Saved threshold: {output_path.name}")
    print(f"Best threshold : {best_threshold:.2f}")
    print(f"Min cost       : {threshold_info['min_cost']:.2f}")
    print(f"Output dir     : {output_dir}")
    print("=" * 70)

    return probabilities, predictions
