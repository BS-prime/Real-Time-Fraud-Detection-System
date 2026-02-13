# ==========================================================================
# --- Import Libraries ---
# ==========================================================================

# Data manupulation
import pandas as pd

# Utils
import re
import yaml
import joblib
from pathlib import Path

# Preprocessing
from sklearn.model_selection import train_test_split, GridSearchCV

# Algo
import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier


# =================================================================================================================
# --- Define the actual function ---
# =================================================================================================================
def model_trainer(
    csv_name: str | None = "fraud_features_seed_42.csv",
    algo_name: str = "xgboost",
) -> tuple:
    """
    Docstring for model_trainer

    :param csv_name: Just put the name of the csv file that is generated from
    the feature engineering step, it should be in the format
    'fraud_features_seed_XX.csv' where XX is the seed number
    used in data generation. The csv file should be located in
    data/features folder.

    :type csv_name: str | None
    :param algo_name: Model key from configs/config.yaml -> models section.
    Example: "xgboost" or "RandomForest".
    :type algo_name: str
    """

    # ----------------------------------------------------------------------------------------------------
    # --- Path Declaration ---
    # ----------------------------------------------------------------------------------------------------

    # Get the csv
    ROOT_DIR = Path(__file__).resolve().parents[2]
    PATH_DIR = ROOT_DIR / "data" / "features" / csv_name

    # Trained model output path
    OUTPUT_DIR = ROOT_DIR / "artifacts" / "models"
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Path to the yaml
    CONFIG_PATH = ROOT_DIR / "configs" / "config.yaml"

    # ----------------------------------------------------------------------------------------------------
    # --- Read the data and extract the seed number ---
    # ----------------------------------------------------------------------------------------------------

    df = pd.read_csv(PATH_DIR)

    # Define the target
    y = df["is_fraud"]

    # Define the features
    X = df.drop(columns=["is_fraud"])

    # Extract the seed number from the csv file
    seed_val = int(re.search(r"_seed_(\d+)", csv_name).group(1))

    # ----------------------------------------------------------------------------------------------------
    # --- 1. Load Config ---
    # ----------------------------------------------------------------------------------------------------

    # Read the config file
    with open(CONFIG_PATH, "r") as f:
        config = yaml.safe_load(f)

    # ----------------------------------------------------------------------------------------------------+
    # --- 2. Map String names to actual Scikit-Learn Classes ---
    # ----------------------------------------------------------------------------------------------------+

    algo_map = {
        "RandomForestClassifier": RandomForestClassifier,
        "XGBClassifier": xgb.XGBClassifier,
    }

    # ----------------------------------------------------------------------------------------------------+
    # --- 3. Data Setup ---
    # ----------------------------------------------------------------------------------------------------+

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    # ----------------------------------------------------------------------------------------------------+
    # --- 4. Train the Selected Model ---
    # ----------------------------------------------------------------------------------------------------+

    # Handling unknown algo error
    if algo_name not in config["models"]:
        available_models = ", ".join(config["models"].keys())
        raise ValueError(
            f"Unknown algo_name '{algo_name}'. Available models: {available_models}"
        )

    settings = config["models"][algo_name]
    model_type = settings["type"]

    
    if model_type not in algo_map:
        available_types = ", ".join(algo_map.keys())
        raise ValueError(
            f"Unsupported model type '{model_type}' for '{algo_name}'. "
            f"Supported types: {available_types}"
        )

    print(f"Running GridSearch for: {algo_name}")

    # Initialize the specific class
    base_model = algo_map[model_type](random_state=seed_val)

    # Setup GridSearch
    grid = GridSearchCV(
        base_model, settings["params"], cv=4, scoring="average_precision"
    )

    # Train the model
    grid.fit(X_train, y_train)

    # Get the best model
    best_model = grid.best_estimator_

    # Saving the model with a unique name
    MODEL_PATH = OUTPUT_DIR / f"{algo_name}_seed_{seed_val}.json"

    # Save the model
    if model_type == "XGBClassifier":
        best_model.save_model(str(MODEL_PATH))
    else:
        joblib.dump(best_model, MODEL_PATH)

    print("=" * 70)
    print(f"Model Name: '{algo_name}_seed_{seed_val}.json'")
    print(f"Best Score: {grid.best_score_:.4f} | Saved to: {OUTPUT_DIR}")
    print("=" * 70)

    return X_test, y_test
