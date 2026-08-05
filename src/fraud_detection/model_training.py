"""Model training utilities."""

import logging
import random
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
import xgboost as xgb
import yaml
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, train_test_split

from fraud_detection.feature_schema import TARGET_COLUMN
from fraud_detection.naming import seed_from_filename
from fraud_detection.paths import (
    CONFIG_PATH,
    FEATURE_DATA_DIR,
    MODEL_DIR,
    create_dir,
)

logger = logging.getLogger(__name__)

MODEL_TYPES = {
    "RandomForestClassifier": RandomForestClassifier,
    "XGBClassifier": xgb.XGBClassifier,
}


class ModelTrainer:
    """
    Train fraud detection models using configured hyperparameters.
    """

    def __init__(
        self,
        config_path: Path = CONFIG_PATH,
        feature_dir: Path = FEATURE_DATA_DIR,
        model_dir: Path = MODEL_DIR,
    ) -> None:
        self.config_path = config_path
        self.feature_dir = feature_dir
        self.model_dir = model_dir
        self.config = self._load_config()

    def _load_config(self) -> dict[str, Any]:
        """
        Read the model training settings from YAML configuration.
        """

        if not self.config_path.exists():
            raise FileNotFoundError(
                f"Training configuration not found: {self.config_path}"
            )

        with self.config_path.open("r", encoding="utf-8") as file:
            return yaml.safe_load(file)

    def _load_training_data(self, csv_name: str) -> pd.DataFrame:
        """
        Load engineered feature data for model training.
        """

        csv_path = self.feature_dir / csv_name
        if not csv_path.exists():
            raise FileNotFoundError(f"Feature file not found: {csv_path}")
        return pd.read_csv(csv_path)

    def _split_features_and_target(
        self, df: pd.DataFrame
    ) -> tuple[pd.DataFrame, pd.Series]:
        """
        Separate feature columns from the target label column.
        """

        return df.drop(columns=[TARGET_COLUMN]), df[TARGET_COLUMN]

    def _build_estimator(self, model_type: str, seed: int) -> Any:
        """
        Construct a classifier instance based on the configured model type.
        """

        if model_type not in MODEL_TYPES:
            supported = ", ".join(MODEL_TYPES)
            raise ValueError(
                f"Unsupported model type '{model_type}'. Supported: {supported}"
            )
        return MODEL_TYPES[model_type](random_state=seed)

    @staticmethod
    def _save_trained_model(model: Any, model_type: str, model_path: Path) -> None:
        """
        Persist the selected model using the correct serializer.
        """

        if model_type == "XGBClassifier":
            model.save_model(str(model_path))
        else:
            joblib.dump(model, model_path)

    def train(
        self, csv_name: str = "fraud_features_seed_42.csv", algo_name: str = "xgboost"
    ) -> tuple[pd.DataFrame, pd.Series]:
        """
        Train the model and save it.
        """

        # 1. Load the feature data and split into features and target.
        seed = seed_from_filename(csv_name)
        df = self._load_training_data(csv_name)
        features, target = self._split_features_and_target(df)

        # 2. Validate the requested algorithm is supported
        if algo_name not in self.config["models"]:
            available = ", ".join(self.config["models"])
            raise ValueError(
                f"Unknown algo_name '{algo_name}'. Available models: {available}"
            )

        # 3. Build the estimator with the specified random seed.
        settings = self.config["models"][algo_name]  # {"type": "XGBClassifier", "params": {...}}
        model_type = settings["type"]  # "XGBClassifier" or "RandomForestClassifier"
        estimator = self._build_estimator(
            model_type, seed
        )  # RandomForestClassifier(random_state=seed) or XGBClassifier(random_state=seed)

        # 4. Split the data into training and testing sets, stratifying by the target label.
        X_train, X_test, y_train, y_test = train_test_split(
            features,
            target,
            test_size=0.2,
            stratify=target,
            random_state=seed,
        )

        # 5. Perform a grid search over the hyperparameter space defined in the configuration.
        logger.info("Running GridSearch for %s", algo_name)
        grid_search = GridSearchCV(
            estimator, settings["params"], scoring="average_precision", n_jobs=-1
        )
        grid_search.fit(X_train, y_train)

        # 6. Save the best model to the artifacts directory.
        output_dir = create_dir(self.model_dir)
        model_path = output_dir / f"{algo_name}_seed_{seed}.json"
        self._save_trained_model(grid_search.best_estimator_, model_type, model_path)

        logger.info(
            "Saved model %s with best score %.4f to %s",
            model_path.name,
            grid_search.best_score_,
            output_dir,
        )

        # Return only the held-out test data so the caller can evaluate model performance.
        return X_test, y_test


def model_trainer(
    csv_name: str = "fraud_features_seed_42.csv", algo_name: str = "xgboost"
) -> tuple[pd.DataFrame, pd.Series]:
    return ModelTrainer().train(csv_name=csv_name, algo_name=algo_name)
