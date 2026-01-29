# ===================================================================================
# --- Import libraries ---
# ===================================================================================

import pandas as pd
import numpy as np
from datetime import datetime
from pathlib import Path

from evidently import Report
from evidently import (
    DataDriftPreset,
    TargetDriftPreset,
    DataQualityPreset
)
from evidently import ColumnMapping



# ===================================================================================
# --- Core Monitoring Function ---
# ===================================================================================

def generate_monitoring_report(
    reference_path: str,
    current_path: str,
    target_col: str | None,
    prediction_col: str | None = None,
    numerical_features: list[str] | None = None,
    categorical_features: list[str] | None = None,
    drift_fail_threshold: float = 0.5,
    output_dir: str = "reports"
):
    """
    CI-friendly monitoring function.

    Returns:
        status: "PASS" | "WARN" | "FAIL"
        drift_score: float
        report_path: str
    """
    
    
    # ===================================================================================
    # --- 1. Load data ---
    # ===================================================================================

    reference = pd.read_csv(reference_path)
    current = pd.read_csv(current_path)

    
    
    # ===================================================================================
    # --- 2. Column Mapping ---
    # ===================================================================================

    column_mapping = ColumnMapping(
        target=target_col,
        prediction=prediction_col,
        numerical_features=numerical_features,
        categorical_features=categorical_features
    )

    
    
    # ===================================================================================
    # --- 3. Decide monitoring mode ---
    # ===================================================================================

    metrics = [
        DataDriftPreset(),
        DataQualityPreset()
    ]

    labels_available = (
        target_col is not None
        and target_col in reference.columns
        and target_col in current.columns
    )

    if labels_available:
        metrics.append(TargetDriftPreset())

    
    
    # ===================================================================================
    # --- 4. Run Evidently ---
    # ===================================================================================

    report = Report(metrics=metrics)

    report.run(
        reference_data=reference,
        current_data=current,
        column_mapping=column_mapping
    )

    
    
    # ===================================================================================
    # --- 5. Persist report (versioned) ---
    # ===================================================================================

    timestamp = datetime.now(datetime.timezone.utc).strftime("%Y%m%d_%H%M%S")
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    report_path = f"{output_dir}/drift_report_{timestamp}.html"
    report.save_html(report_path)

    
    
    # ===================================================================================
    # --- 6. Extract drift signal (machine-readable) ---
    # ===================================================================================

    report_dict = report.as_dict()

    drifted_features = [
        m for m in report_dict["metrics"]
        if "drift_detected" in m.get("result", {})
        and m["result"]["drift_detected"] is True
    ]

    total_checked = len([
        m for m in report_dict["metrics"]
        if "drift_detected" in m.get("result", {})
    ])

    drift_score = (
        len(drifted_features) / total_checked
        if total_checked > 0
        else 0.0
    )

    
    
    # ===================================================================================
    # --- 7. Decide CI status ---
    # ===================================================================================

    if drift_score >= drift_fail_threshold:
        status = "FAIL"
    elif drift_score >= drift_fail_threshold * 0.6:
        status = "WARN"
    else:
        status = "PASS"

    
    
    # ===================================================================================
    # --- 8. Final output ---
    # ===================================================================================

    print("====================================")
    print(f"Monitoring status : {status}")
    print(f"Drift score       : {drift_score:.2f}")
    print(f"Report saved at   : {report_path}")
    print("====================================")

    return {
        "status": status,
        "drift_score": drift_score,
        "report_path": report_path
    }
