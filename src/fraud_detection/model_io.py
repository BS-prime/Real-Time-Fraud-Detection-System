"""Model loading, prediction, and threshold helpers."""

from pathlib import Path
from typing import Any
import json

import joblib
import numpy as np
import pandas as pd
import xgboost as xgb

from fraud_detection.feature_schema import FEATURE_COLUMNS


def load_model(model_path: str | Path) -> Any:
    """
    Load either a joblib scikit-learn model or a native XGBoost model.
    """

    model_path = Path(model_path)
    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")

    joblib_error: Exception | None = None
    try:
        return joblib.load(model_path)
    except Exception as error:
        joblib_error = error

    booster = xgb.Booster()
    try:
        booster.load_model(str(model_path))
        return booster
    except Exception as xgboost_error:
        raise RuntimeError(
            f"Could not load model from {model_path}. "
            f"Joblib failed with: {joblib_error}"
        ) from xgboost_error


def predict_fraud_probability(
    model: Any, features: pd.DataFrame | np.ndarray
) -> np.ndarray:
    """
    Return fraud probabilities for models with a common interface.
    """

    if hasattr(model, "predict_proba"):
        return np.asarray(model.predict_proba(features)[:, 1], dtype=float)

    if isinstance(model, xgb.Booster):
        feature_names = None
        if isinstance(features, pd.DataFrame):
            feature_names = [str(column) for column in features.columns]
        return np.asarray(
            model.predict(xgb.DMatrix(features, feature_names=feature_names)),
            dtype=float,
        )

    return np.asarray(model.predict(features), dtype=float)


def model_feature_names(model: Any) -> list[str] | None:
    """
    Read feature names from a model when the model stores them.
    """

    if hasattr(model, "feature_names_in_"):
        return [str(column) for column in model.feature_names_in_]

    if isinstance(model, xgb.Booster) and model.feature_names:
        return [str(column) for column in model.feature_names]

    return None


def align_features_to_model(
    features: pd.DataFrame,
    model: Any,
    fallback_columns: list[str] | None = None,
) -> pd.DataFrame:
    """
    Return columns in the order expected by the model.
    """

    expected_columns = model_feature_names(model) or fallback_columns or FEATURE_COLUMNS
    aligned = features.copy()
    aligned.columns = aligned.columns.astype(str)

    for column in expected_columns:
        if column not in aligned.columns:
            aligned[column] = 0

    return aligned.reindex(columns=expected_columns, fill_value=0)


def load_threshold(threshold_path: str | Path) -> float:
    """
    Load the optimized decision threshold from JSON metadata.
    """

    threshold_path = Path(threshold_path)
    if not threshold_path.exists():
        raise FileNotFoundError(f"Threshold file not found: {threshold_path}")

    with threshold_path.open("r", encoding="utf-8") as file:
        return float(json.load(file)["best_threshold"])


def save_json(data: dict[str, Any], path: str | Path) -> None:
    """
    Write small JSON metadata files with a stable, readable format."""
    
    with Path(path).open("w", encoding="utf-8") as file:
        json.dump(data, file, indent=2)
