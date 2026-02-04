# ===================================================================================
# --- Imports ---
# ===================================================================================

import pandas as pd
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
# --- Defining path ---
# ===================================================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPORTS_DIR = PROJECT_ROOT / "reports" / "drift_monitoring"
REPORTS_DIR.mkdir(parents=True, exist_ok=True)


# ===================================================================================
# --- Core Monitoring Function ---
# ===================================================================================

def generate_monitoring_report(
    model,
    X_train,
    X_test,
    y_train=None,
    y_test=None,
    numerical_features=None,
    categorical_features=None,
    drift_fail_threshold=0.5,
    output_dir=REPORTS_DIR,
):
    
    '''
    Generate a monitoring report using Evidently.
    '''


    # ==================================================================================
    # --- 1. Normalize inputs ---
    # ==================================================================================

    X_train = pd.DataFrame(X_train).copy()
    X_test = pd.DataFrame(X_test).copy()

    X_train.columns = X_train.columns.astype(str)
    X_test.columns = X_test.columns.astype(str)

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)



    # ==================================================================================
    # --- 2. Build reference & current datasets ---
    # ==================================================================================

    reference = X_train.copy()
    current = X_test.copy()

    target_col = "__target__" if y_train is not None and y_test is not None else None
    prediction_col = "__prediction__"

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

    column_mapping = ColumnMapping(
        target=target_col,
        numerical_features=numerical_features,
        categorical_features=categorical_features,
        prediction=prediction_col,
    )



    # ==================================================================================
    # --- 4. Metrics ---
    # ==================================================================================

    metrics = [
        DataDriftPreset(),
        DataQualityPreset(),
    ]

    if target_col:
        metrics.append(TargetDriftPreset())



    # ==================================================================================
    # --- 5. Run Evidently ---
    # ==================================================================================

    report = Report(metrics=metrics)

    report.run(
        reference_data=reference,
        current_data=current,
        column_mapping=column_mapping,
    )



    # ==================================================================================
    # --- 6. Save report (HTML) ---
    # ==================================================================================

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    report_path = output_dir / f"drift_report_{timestamp}.html"

    report.save_html(report_path.as_posix())

    assert report_path.exists(), "HTML report was NOT written"



    # ==================================================================================
    # --- 7. Drift score ---
    # ==================================================================================

    report_dict = report.as_dict()

    drift_flags = [
        m["result"]["drift_detected"]
        for m in report_dict.get("metrics", [])
        if isinstance(m.get("result"), dict)
        and "drift_detected" in m["result"]
    ]

    drift_score = sum(drift_flags) / len(drift_flags) if drift_flags else 0.0



    # ==================================================================================
    # --- 8. Status ---
    # ==================================================================================

    if drift_score >= drift_fail_threshold:
        status = "FAIL"
    elif drift_score >= drift_fail_threshold * 0.6:
        status = "WARN"
    else:
        status = "PASS"

    print("====================================")
    print("Train–Test Drift Monitoring")
    print(f"Status      : {status}")
    print(f"Drift score : {drift_score:.2f}")
    print(f"Report      : {report_path.resolve()}")
    print("====================================")

    return {
        "status": status,
        "drift_score": drift_score,
        "report_path": str(report_path.resolve()),
    }
