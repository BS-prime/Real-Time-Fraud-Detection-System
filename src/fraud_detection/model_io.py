"""Model loading, prediction, and threshold helpers."""

import json
import logging
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import xgboost as xgb

from fraud_detection.feature_schema import FEATURE_COLUMNS

logger = logging.getLogger(__name__)


class ModelIO:
    """
    Encapsulate model serialization and inference compatibility.
    """

    @staticmethod
    def load_model(model_path: str | Path) -> Any:
        """
        Load a saved model from disk, supporting joblib and XGBoost formats.
        """

        model_path = Path(model_path)
        if not model_path.exists():
            raise FileNotFoundError(f"Model file not found: {model_path}")

        try:
            return joblib.load(model_path)
        except Exception as joblib_error:
            logger.debug("Joblib load failed for %s: %s", model_path, joblib_error)

        booster = xgb.Booster()
        try:
            booster.load_model(str(model_path))
            return booster
        except Exception as xgboost_error:
            raise RuntimeError(
                f"Could not load model from {model_path}. "
                f"Joblib failed with: {joblib_error}"
            ) from xgboost_error

    @staticmethod
    def predict_fraud_probability(
        model: Any, features: pd.DataFrame | np.ndarray
    ) -> np.ndarray:
        """
        Return fraud probability estimates for the positive class.
        """
        
        if hasattr(model, "predict_proba"):
            return np.asarray(model.predict_proba(features)[:, 1], dtype=float)

        if isinstance(model, xgb.Booster):
            feature_names = (
                [str(column) for column in features.columns]
                if isinstance(features, pd.DataFrame)
                else None
            )
            return np.asarray(
                model.predict(xgb.DMatrix(features, feature_names=feature_names)),
                dtype=float,
            )

        return np.asarray(model.predict(features), dtype=float)

    @staticmethod
    def model_feature_names(model: Any) -> list[str] | None:
        """
        Retrieve the feature names expected by the model.
        """
        
        if hasattr(model, "feature_names_in_"):
            return [str(column) for column in model.feature_names_in_]

        if isinstance(model, xgb.Booster) and model.feature_names:
            return [str(column) for column in model.feature_names]

        return None

    @classmethod
    def align_features_to_model(
        cls,
        features: pd.DataFrame,
        model: Any,
        fallback_columns: list[str] | None = None,
    ) -> pd.DataFrame:
        """
        Ensure the feature matrix matches the model's expected input schema.
        """

        expected_columns = (
            cls.model_feature_names(model) or fallback_columns or FEATURE_COLUMNS
        )
        aligned = features.copy()
        aligned.columns = aligned.columns.astype(str)

        for column in expected_columns:
            if column not in aligned.columns:
                aligned[column] = 0

        return aligned.reindex(columns=expected_columns, fill_value=0)

    @staticmethod
    def load_threshold(threshold_path: str | Path) -> float:
        """
        Load the optimal threshold for fraud detection.
        """
        threshold_path = Path(threshold_path)
        if not threshold_path.exists():
            raise FileNotFoundError(f"Threshold file not found: {threshold_path}")

        with threshold_path.open("r", encoding="utf-8") as file:
            return float(json.load(file)["best_threshold"])

    @staticmethod
    def save_json(data: dict[str, Any], path: str | Path) -> None:
        """
        Save data to a JSON file.
        """

        with Path(path).open("w", encoding="utf-8") as file:
            json.dump(data, file, indent=2)


def load_model(model_path: str | Path) -> Any:
    return ModelIO.load_model(model_path)


def predict_fraud_probability(
    model: Any, features: pd.DataFrame | np.ndarray
) -> np.ndarray:
    return ModelIO.predict_fraud_probability(model, features)


def model_feature_names(model: Any) -> list[str] | None:
    return ModelIO.model_feature_names(model)


def align_features_to_model(
    features: pd.DataFrame,
    model: Any,
    fallback_columns: list[str] | None = None,
) -> pd.DataFrame:
    return ModelIO.align_features_to_model(
        features, model, fallback_columns=fallback_columns
    )


def load_threshold(threshold_path: str | Path) -> float:
    return ModelIO.load_threshold(threshold_path)


def save_json(data: dict[str, Any], path: str | Path) -> None:
    return ModelIO.save_json(data, path)
