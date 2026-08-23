"""
Shared filesystem locations used across the project.
"""

from pathlib import Path

PROJECT_ROOT: Path = Path(__file__).resolve().parents[2]

DATA_DIR: Path = PROJECT_ROOT / "data"
SIMULATED_DATA_DIR: Path = DATA_DIR / "simulated"
FEATURE_DATA_DIR: Path = DATA_DIR / "features"

ARTIFACTS_DIR: Path = PROJECT_ROOT / "artifacts"
MODEL_DIR: Path = ARTIFACTS_DIR / "models"
THRESHOLD_DIR: Path = ARTIFACTS_DIR / "model_thresholds"
SAVED_MODELS_PATH: Path = ARTIFACTS_DIR / "model_paths"

REPORTS_DIR: Path = PROJECT_ROOT / "reports"
HYPERPARAMS_CONFIG_PATH: Path = PROJECT_ROOT / "configs" / "hyperparams.yaml"
PARAMS_CONFIG_PATH: Path = PROJECT_ROOT / "configs" / "params.yaml"

def create_dir(path: Path) -> Path:
    """
    Create a directory if it does not already exist.

    This helper is used throughout the project to ensure artifact and data
    directories are available before writing files.
    """

    path.mkdir(parents=True, exist_ok=True)
    return path


def simulated_transactions_path(seed: int | str) -> Path:
    """
    Return the standard raw transaction CSV path for a seed.
    """

    return SIMULATED_DATA_DIR / f"simulated_transactions_seed_{seed}.csv"


def train_feature_file_path(seed: int | str) -> Path:
    """
    Return the standard engineered feature CSV path for a seed.
    """

    return FEATURE_DATA_DIR / f"fraud_features_train_seed_{seed}.csv"

def test_feature_file_path(seed: int | str) -> Path:
    """
    Return the standard engineered feature CSV path for a seed.
    """

    return FEATURE_DATA_DIR / f"fraud_features_test_seed_{seed}.csv"
