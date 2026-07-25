"""
Batch drift monitoring with Evidently.
"""

from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
from evidently import ColumnMapping
from evidently.metric_preset import (
    DataDriftPreset,
    DataQualityPreset,
    TargetDriftPreset,
)
from evidently.report import Report

from fraud_detection.feature_engineering import feature_engineer
from fraud_detection.feature_schema import TARGET_COLUMN
from fraud_detection.model_io import (
    align_features_to_model,
    load_model,
    predict_fraud_probability,
)
from fraud_detection.naming import seed_from_filename
from fraud_detection.paths import (
    FEATURE_DATA_DIR,
    MODEL_DIR,
    REPORTS_DIR,
    SIMULATED_DATA_DIR,
)
from fraud_detection.paths import ensure_directory


def load_model_ready_dataset(dataset_name: str) -> pd.DataFrame:
    """
    Load an engineered feature file, or engineer it from raw transactions.
    """

    feature_path = FEATURE_DATA_DIR / dataset_name
    if feature_path.exists():
        return pd.read_csv(feature_path)

    seed = seed_from_filename(dataset_name)
    derived_feature_path = FEATURE_DATA_DIR / f"fraud_features_seed_{seed}.csv"
    if derived_feature_path.exists():
        return pd.read_csv(derived_feature_path)

    simulated_path = SIMULATED_DATA_DIR / dataset_name
    if simulated_path.exists():
        return feature_engineer(dataset_name)

    raise FileNotFoundError(
        f"Could not find '{dataset_name}' in data/features or data/simulated."
    )


def split_dataset(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Return features and target for a model-ready dataset."""
    return df.drop(columns=[TARGET_COLUMN]), df[TARGET_COLUMN]


def drift_status(drift_score: float, fail_threshold: float) -> str:
    """Translate a numeric drift score into a plain-language status."""
    if drift_score >= fail_threshold:
        return "CRITICAL: retraining recommended"
    if drift_score >= fail_threshold * 0.6:
        return "WARNING: review drift report"
    return "OK: no action needed"


def _drift_score(report: Report) -> float:
    report_dict = report.as_dict()
    drift_flags = [
        metric["result"]["drift_detected"]
        for metric in report_dict.get("metrics", [])
        if isinstance(metric.get("result"), dict)
        and "drift_detected" in metric["result"]
    ]
    return sum(drift_flags) / len(drift_flags) if drift_flags else 0.0


def generate_monitoring_report(
    model_name: str,
    trained_dataset: str,
    new_dataset: str,
    drift_fail_threshold: float | None = 0.5,
) -> dict[str, float | str]:
    """
    Compare reference and current datasets, then save an Evidently HTML report.
    """

    fail_threshold = 0.5 if drift_fail_threshold is None else drift_fail_threshold
    model = load_model(MODEL_DIR / model_name)

    reference_df = load_model_ready_dataset(trained_dataset)
    current_df = load_model_ready_dataset(new_dataset)

    X_reference, y_reference = split_dataset(reference_df)
    X_current, y_current = split_dataset(current_df)
    X_reference = align_features_to_model(
        X_reference,
        model,
        fallback_columns=list(X_reference.columns),
    )
    X_current = align_features_to_model(
        X_current,
        model,
        fallback_columns=list(X_reference.columns),
    )

    reference = X_reference.copy()
    current = X_current.copy()
    target_col = "__target__"
    prediction_col = "__prediction__"

    reference[target_col] = y_reference.values
    current[target_col] = y_current.values
    reference[prediction_col] = predict_fraud_probability(model, X_reference)
    current[prediction_col] = predict_fraud_probability(model, X_current)

    numerical_features = X_reference.select_dtypes(include=["number"]).columns.tolist()
    categorical_features = X_reference.select_dtypes(
        include=["object", "category", "bool"]
    ).columns.tolist()

    report = Report(
        metrics=[
            DataDriftPreset(),
            DataQualityPreset(),
            TargetDriftPreset(),
        ]
    )
    report.run(
        reference_data=reference,
        current_data=current,
        column_mapping=ColumnMapping(
            target=target_col,
            prediction=prediction_col,
            numerical_features=numerical_features,
            categorical_features=categorical_features,
        ),
    )

    reports_dir = ensure_directory(REPORTS_DIR / "drift_monitoring")
    ref_version = Path(trained_dataset).stem
    current_version = Path(new_dataset).stem
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    report_path = (
        reports_dir
        / f"drift_report_{current_version}_vs_{ref_version}_{timestamp}.html"
    )
    report.save_html(report_path.as_posix())

    score = _drift_score(report)
    status = drift_status(score, fail_threshold)

    print("=" * 70)
    print("Data and target drift monitoring")
    print(f"Status     : {status}")
    print(f"Drift score: {score:.2f}")
    print(f"Report     : {report_path.resolve()}")
    print("=" * 70)

    return {
        "status": status,
        "drift_score": score,
        "report_path": str(report_path.resolve()),
    }
