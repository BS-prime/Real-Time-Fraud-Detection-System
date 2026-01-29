# ===================================================================================
# --- Import libraries ---
# ===================================================================================

import pandas as pd
import numpy as np
from datetime import datetime, timezone
from pathlib import Path

from evidently.report import Report
from evidently import ColumnMapping

from evidently.metric_preset import (
    DataDriftPreset,
    DataQualityPreset,
    TargetDriftPreset
)

# ===================================================================================
# --- Core Monitoring Function ---
# ===================================================================================

# Define project root to manage paths
PACKAGE_ROOT = Path(__file__).resolve().parent          
PROJECT_ROOT = PACKAGE_ROOT.parents[1]                 
REPORTS_DIR = PROJECT_ROOT / "reports"

def generate_monitoring_report(
    model,
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series | None = None,
    y_test: pd.Series | None = None,
    drift_fail_threshold: float = 0.5,
    output_dir: str | Path = REPORTS_DIR
):
    """
    Drift check including prediction drift.
    """
    # ===========================================================================
    # --- 0. Path ---
    # ===========================================================================
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    
    
    # ===========================================================================
    # --- 1. Build reference & current data ---
    # ===========================================================================
    
    reference = X_train.copy()
    current = X_test.copy()

    target_col = None
    prediction_col = "__prediction__"

    if y_train is not None and y_test is not None:
        target_col = "__target__"
        reference[target_col] = y_train.values
        current[target_col] = y_test.values

    # Predictions
    if hasattr(model, "predict_proba"):
        reference[prediction_col] = model.predict_proba(X_train)[:, 1]
        current[prediction_col] = model.predict_proba(X_test)[:, 1]
    else:
        reference[prediction_col] = model.predict(X_train)
        current[prediction_col] = model.predict(X_test)

    
    
    # ===========================================================================
    # --- 2. Column mapping ---
    # ===========================================================================
    try:
        if len(model.feature_names_in_) == X_train.shape[1] and len(model.feature_names_in_) == X_test.shape[1]:          
            
            numerical_features = pd.DataFrame(X_train).select_dtypes(include=['int64', 'float64']).columns.tolist()
            categorical_features = pd.DataFrame(X_train).select_dtypes(include=['object', 'category', 'bool']).columns.tolist()
    
    except AttributeError:
        raise AttributeError("Feature names not found in the model. Please ensure the model is trained with feature names.")
    
    column_mapping = ColumnMapping(
        target=target_col,
        prediction=prediction_col,
        numerical_features=numerical_features,
        categorical_features=categorical_features
    )

    
    
    # ===========================================================================
    # --- 3. Metrics ---
    # ===========================================================================
    
    metrics = [
        DataDriftPreset(),
        DataQualityPreset(),
        TargetDriftPreset() if target_col else None
    ]
    metrics = [m for m in metrics if m is not None]

    
    
    # ===========================================================================
    # --- 4. Run Evidently ---
    # ===========================================================================
    
    report = Report(metrics=metrics)
    report.run(
        reference_data=reference,
        current_data=current,
        column_mapping=column_mapping
    )

    
    
    # ===========================================================================
    # --- 5. Save report ---
    # ===========================================================================
    
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    report_path = f"{output_dir}/drift_report_{timestamp}.html"
    report.save_html(report_path)

    
    
    # ===========================================================================
    # --- 6. Extract drift score ---
    # ===========================================================================
    
    report_dict = report.as_dict()

    checked = [
        m for m in report_dict["metrics"]
        if "drift_detected" in m.get("result", {})
    ]

    drifted = [
        m for m in checked
        if m["result"]["drift_detected"] is True
    ]

    drift_score = len(drifted) / len(checked) if checked else 0.0

    
    
    # ===========================================================================
    # --- 7. Status ---
    # ===========================================================================
    
    if drift_score >= drift_fail_threshold:
        status = "FAIL"
    elif drift_score >= drift_fail_threshold * 0.6:
        status = "WARN"
    else:
        status = "PASS"

    print("====================================")
    print("Train–Test Drift (with Prediction)")
    print(f"Status      : {status}")
    print(f"Drift score : {drift_score:.2f}")
    print(f"Report      : {report_path}")
    print("====================================")

    return {
        "status": status,
        "drift_score": drift_score,
        "report_path": report_path
    }
