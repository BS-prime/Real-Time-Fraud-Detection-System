# ======================================================================================
# --- Imports ---
# ======================================================================================

import joblib
import pandas as pd
from datetime import datetime, timezone
from pathlib import Path

import xgboost as xgb

from evidently.report import Report
from evidently import ColumnMapping
from evidently.metric_preset import (
    DataDriftPreset,
    DataQualityPreset,
    TargetDriftPreset,
)


# ======================================================================================
# --- Core Monitoring Function ---
# ======================================================================================


def generate_monitoring_report(
    model_name: str,
    trained_dataset: str,
    new_dataset: str,
    drift_fail_threshold: float | None = 0.5,
):
    # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    # --- 0. Defining path ---
    # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

    # Locate the project root
    PROJECT_ROOT = Path(__file__).resolve().parents[2]

    # Create the output directory
    REPORTS_DIR = PROJECT_ROOT / "reports" / "drift_monitoring"
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    # Locate the csv files
    CURRENT_PATH = PROJECT_ROOT / "data" / "simulated" / new_dataset
    REFERENCE_PATH = PROJECT_ROOT / "data" / "simulated" / trained_dataset

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

    # Put the csv files into DataFrames
    df_current = pd.read_csv(CURRENT_PATH)
    df_reference = pd.read_csv(REFERENCE_PATH)

    # Selecting the features
    X_train = df_current.drop(columns=["is_fraud"])
    X_test = df_reference.drop(columns=["is_fraud"])

    # Selecting the target
    y_train = df_current["is_fraud"]
    y_test = df_reference["is_fraud"]

    # Get the column names
    X_train.columns = X_train.columns.astype(str)
    X_test.columns = X_test.columns.astype(str)

    # Separating the numerical and categorical features
    numerical_features = X_train.select_dtypes(
        include=["int64", "float64"]
    ).columns.tolist()
    categorical_features = X_train.select_dtypes(
        include=["object", "category", "bool"]
    ).columns.tolist()

    # ==================================================================================
    # --- 2. Inclusion of target and prediction columns ---
    # ==================================================================================

    # Copy the features to a new DataFrame
    reference = X_train.copy()
    current = X_test.copy()

    # Create empty columns for target and prediction
    target_col = "__target__"
    prediction_col = "__prediction__"

    # Fill the empty columns
    if target_col:
        reference[target_col] = y_train.values
        current[target_col] = y_test.values

    if hasattr(model, "predict_proba"):
        reference[prediction_col] = model.predict_proba(X_train)[:, 1]
        current[prediction_col] = model.predict_proba(X_test)[:, 1]
    else:
        reference[prediction_col] = model.predict(X_train)
        current[prediction_col] = model.predict(X_test)

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
    ref_ver = Path(REFERENCE_PATH).stem

    # Same thing for current
    curr_ver = Path(CURRENT_PATH).stem

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
