"""Model training utilities."""

from typing import Any

import pandas as pd
import xgboost as xgb
import yaml
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, train_test_split

from fraud_detection.feature_schema import TARGET_COLUMN
from fraud_detection.naming import seed_from_filename
from fraud_detection.paths import (
    CONFIG_PATH,
    FEATURE_DATA_DIR,
    MODEL_DIR,
    ensure_directory,
)

MODEL_TYPES = {
    "RandomForestClassifier": RandomForestClassifier,
    "XGBClassifier": xgb.XGBClassifier,
}


def load_training_data(csv_name: str) -> pd.DataFrame:

    """
    Load a model-ready feature file from ``data/features``.
    """

    csv_path = FEATURE_DATA_DIR / csv_name
    if not csv_path.exists():
        raise FileNotFoundError(f"Feature file not found: {csv_path}")
    return pd.read_csv(csv_path)


def load_config() -> dict[str, Any]:
    """
    Read the YAML training configuration.
    """

    with CONFIG_PATH.open("r", encoding="utf-8") as file:
        return yaml.safe_load(file)


def split_features_and_target(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """
    Separate model inputs from the fraud label.
    """

    return df.drop(columns=[TARGET_COLUMN]), df[TARGET_COLUMN]


def build_estimator(model_type: str, seed: int):
    """
    Create a supported estimator from its config name.
    """

    if model_type not in MODEL_TYPES:
        supported = ", ".join(MODEL_TYPES)
        raise ValueError(
            f"Unsupported model type '{model_type}'. Supported: {supported}"
        )
    return MODEL_TYPES[model_type](random_state=seed)


def save_trained_model(model, model_type: str, model_path) -> None:
    """
    Persist a trained model using the format expected by its library."""

    if model_type == "XGBClassifier":
        model.save_model(str(model_path))
    else:
        joblib.dump(model, model_path)


def model_trainer(
    csv_name: str = "fraud_features_seed_42.csv",
    algo_name: str = "xgboost",
) -> tuple[pd.DataFrame, pd.Series]:
    """
    Train the configured model and return the holdout test split.
    """
    
    seed = seed_from_filename(csv_name)
    df = load_training_data(csv_name)
    features, target = split_features_and_target(df)
    config = load_config()

    if algo_name not in config["models"]:
        available = ", ".join(config["models"])
        raise ValueError(
            f"Unknown algo_name '{algo_name}'. Available models: {available}"
        )

    settings = config["models"][algo_name]
    model_type = settings["type"]
    estimator = build_estimator(model_type, seed)

    X_train, X_test, y_train, y_test = train_test_split(
        features,
        target,
        test_size=0.2,
        stratify=target,
        random_state=seed,
    )

    print(f"Running GridSearch for: {algo_name}")
    grid_search = GridSearchCV(
        estimator,
        settings["params"],
        cv=4,
        scoring="average_precision",
    )
    grid_search.fit(X_train, y_train)

    output_dir = ensure_directory(MODEL_DIR)
    model_path = output_dir / f"{algo_name}_seed_{seed}.json"
    save_trained_model(grid_search.best_estimator_, model_type, model_path)

    print("=" * 70)
    print(f"Saved model: {model_path.name}")
    print(f"Best score : {grid_search.best_score_:.4f}")
    print(f"Output dir : {output_dir}")
    print("=" * 70)

    return X_test, y_test
