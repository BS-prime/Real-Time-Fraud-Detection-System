# ======================================================================================
# --- Imports ---
# ======================================================================================

import joblib
import pandas as pd
from datetime import datetime, timezone
from pathlib import Path
import re

import xgboost as xgb

from evidently.report import Report
from evidently import ColumnMapping
from evidently.metric_preset import (
    DataDriftPreset,
    DataQualityPreset,
    TargetDriftPreset,
)
from .feature_engineering import feature_engineering


# ======================================================================================
# --- Core Monitoring Function ---
# ======================================================================================


def generate_monitoring_report(
    model_name: str,
    trained_dataset: str,
    new_dataset: str,
    drift_fail_threshold: float | None = 0.5,
):
    
    '''
    Docstring for generate_monitoring_report
    
    :param model_name: Just pick a model for which you want check drift
    :type model_name: str
    :param trained_dataset: The dataset in which current model is trained on
    :type trained_dataset: str
    :param new_dataset: The new dataset
    :type new_dataset: str
    :param drift_fail_threshold: Just a number
    :type drift_fail_threshold: float | None
    '''
    
    # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    # --- Defining some helper functions ---
    # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

    def _seed_from_name(name: str) -> str | None:
        match = re.search(r"_seed_(\d+)", name)
        return match.group(1) if match else None

    # This function check for the new dat
    def _load_model_ready_dataset(dataset_name: str, project_root: Path) -> pd.DataFrame:
        # User might pass a pre-engineered file directly.
        feature_path = project_root / "data" / "features" / dataset_name
        if feature_path.exists():
            return pd.read_csv(feature_path)

        # If a simulated file is passed, first try corresponding engineered file.
        seed = _seed_from_name(dataset_name)
        if seed:
            derived_name = f"fraud_features_seed_{seed}.csv"
            derived_path = project_root / "data" / "features" / derived_name
            if derived_path.exists():
                return pd.read_csv(derived_path)

        # Fallback: engineer features from raw simulated data.
        simulated_path = project_root / "data" / "simulated" / dataset_name
        if simulated_path.exists():
            return feature_engineering(dataset_name)

        raise FileNotFoundError(
            f"Could not find '{dataset_name}' in data/features or data/simulated."
        )

    def _model_feature_names(model) -> list[str] | None:
        if hasattr(model, "feature_names_in_"):
            return [str(col) for col in model.feature_names_in_]
        if isinstance(model, xgb.Booster) and model.feature_names:
            return [str(col) for col in model.feature_names]
        return None

    def _align_features_to_model(X: pd.DataFrame, model) -> pd.DataFrame:
        expected = _model_feature_names(model)
        X = X.copy()
        X.columns = X.columns.astype(str)

        if not expected:
            return X

        for col in expected:
            if col not in X.columns:
                X[col] = 0

        return X.reindex(columns=expected, fill_value=0)

    # Add a model-agnostic prediction helper
    def _predict_scores(model, X: pd.DataFrame):
        if hasattr(model, "predict_proba"):
            return model.predict_proba(X)[:, 1]
        if isinstance(model, xgb.Booster):
            return model.predict(xgb.DMatrix(X))
        return model.predict(X)

    # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    # --- 0. Defining path ---
    # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

    # Locate the project root
    PROJECT_ROOT = Path(__file__).resolve().parents[2]

    # Create the output directory
    REPORTS_DIR = PROJECT_ROOT / "reports" / "drift_monitoring"
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    # Model Path
    MODEL_PATH = PROJECT_ROOT / "artifacts" / "models" / model_name

    # Load model
    try:
        model = joblib.load(MODEL_PATH)
    except:
        model = xgb.Booster()
        model.load_model(str(MODEL_PATH))

    # ==================================================================================
    # --- 1. Normalize inputs ---
    # ==================================================================================

    # Load model-ready datasets.
    df_reference = _load_model_ready_dataset(trained_dataset, PROJECT_ROOT)
    df_current = _load_model_ready_dataset(new_dataset, PROJECT_ROOT)

    # Selecting the features
    X_reference = df_reference.drop(columns=["is_fraud"])
    X_current = df_current.drop(columns=["is_fraud"])

    # Selecting the target
    y_reference = df_reference["is_fraud"]
    y_current = df_current["is_fraud"]

    # Get the column names
    X_reference.columns = X_reference.columns.astype(str)
    X_current.columns = X_current.columns.astype(str)
    X_reference = _align_features_to_model(X_reference, model)
    X_current = _align_features_to_model(X_current, model)

    # Separating the numerical and categorical features
    numerical_features = X_reference.select_dtypes(
        include=["int64", "float64"]
    ).columns.tolist()
    categorical_features = X_reference.select_dtypes(
        include=["object", "category", "bool"]
    ).columns.tolist()

    # ==================================================================================
    # --- 2. Inclusion of target and prediction columns ---
    # ==================================================================================

    # Copy the features to a new DataFrame
    reference = X_reference.copy()
    current = X_current.copy()

    # Create empty columns for target and prediction
    target_col = "__target__"
    prediction_col = "__prediction__"

    # Fill the empty columns
    if target_col:
        reference[target_col] = y_reference.values
        current[target_col] = y_current.values

    reference[prediction_col] = _predict_scores(model, X_reference)
    current[prediction_col] = _predict_scores(model, X_current)

    # ==================================================================================
    # --- 3. Column mapping ---
    # ==================================================================================

    # Let evidently know what type of columns he is dealing with
    column_mapping = ColumnMapping(
        target=target_col,
        numerical_features=numerical_features,
        categorical_features=categorical_features,
        prediction=prediction_col,
    )

    # ==================================================================================
    # --- 4. Metrics ---
    # ==================================================================================

    # Let evidently know what it should measure
    metrics = [DataDriftPreset(), DataQualityPreset(), TargetDriftPreset()]

    # ==================================================================================
    # --- 5. Run Evidently ---
    # ==================================================================================

    # Create a nice report :)
    report = Report(metrics=metrics)

    report.run(
        reference_data=reference,
        current_data=current,
        column_mapping=column_mapping,
    )

    # ==================================================================================
    # --- 6. Save report (HTML) ---
    # ==================================================================================

    # Preserving version names for clearity
    ref_ver = Path(trained_dataset).stem

    # Same thing for current
    curr_ver = Path(new_dataset).stem

    # Define the report path with timestamps
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    report_path = REPORTS_DIR / f"drift_report_{curr_ver}_vs_{ref_ver}_{timestamp}.html"

    # Save the report
    report.save_html(report_path.as_posix())

    # Verify the report was written
    assert report_path.exists(), "HTML report was NOT written"

    # ==================================================================================
    # --- 7. Drift score ---
    # ==================================================================================

    report_dict = report.as_dict()

    drift_flags = [
        m["result"]["drift_detected"]
        for m in report_dict.get("metrics", [])
        if isinstance(m.get("result"), dict) and "drift_detected" in m["result"]
    ]

    drift_score = sum(drift_flags) / len(drift_flags) if drift_flags else 0.0

    # ==================================================================================
    # --- 8. Status ---
    # ==================================================================================

    if drift_score >= drift_fail_threshold:
        status = "CRITICAL FAILURE!!! RETRAIN IMMEDIATELY"
    elif drift_score >= drift_fail_threshold * 0.6:
        status = "WARNING!!! attention required."
    else:
        status = "No action needed!!!"

    print("====================================")
    print("Data and Target Drift Monitoring")
    print(f"Status      : {status}")
    print(f"Drift score : {drift_score:.2f}")
    print(f"Report      : {report_path.resolve()}")
    print("====================================")

    return {
        "status": status,
        "drift_score": drift_score,
        "report_path": str(report_path.resolve()),
    }
