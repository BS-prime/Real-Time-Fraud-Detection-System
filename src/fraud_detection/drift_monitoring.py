"""
Batch drift monitoring with Evidently.
"""

import logging
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
    ensure_directory,
)

logger = logging.getLogger(__name__)


class DriftMonitor:
    """Generate drift monitoring reports and translate drift scores."""

    def __init__(
        self,
        model_dir: Path = MODEL_DIR,
        feature_dir: Path = FEATURE_DATA_DIR,
        simulated_dir: Path = SIMULATED_DATA_DIR,
        reports_dir: Path = REPORTS_DIR,
    ) -> None:
        self.model_dir = model_dir
        self.feature_dir = feature_dir
        self.simulated_dir = simulated_dir
        self.reports_dir = reports_dir

    def load_model_ready_dataset(self, dataset_name: str) -> pd.DataFrame:
        """Load a dataset that is already feature-engineered or derive features as needed."""
        feature_path = self.feature_dir / dataset_name
        if feature_path.exists():
            return pd.read_csv(feature_path)

        seed = seed_from_filename(dataset_name)
        derived_feature_path = self.feature_dir / f"fraud_features_seed_{seed}.csv"
        if derived_feature_path.exists():
            return pd.read_csv(derived_feature_path)

        simulated_path = self.simulated_dir / dataset_name
        if simulated_path.exists():
            return feature_engineer(dataset_name)

        raise FileNotFoundError(
            f"Could not find '{dataset_name}' in {self.feature_dir} or {self.simulated_dir}."
        )

    @staticmethod
    def split_dataset(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
        """Split a dataset into features and the fraud target label."""
        return df.drop(columns=[TARGET_COLUMN]), df[TARGET_COLUMN]

    @staticmethod
    def drift_status(drift_score: float, fail_threshold: float) -> str:
        """Convert a normalized drift score into an actionable status string."""
        if drift_score >= fail_threshold:
            return "CRITICAL: retraining recommended"
        if drift_score >= fail_threshold * 0.6:
            return "WARNING: review drift report"
        return "OK: no action needed"

    @staticmethod
    def _drift_score(report: Report) -> float:
        """Compute a simple drift score from the Evidently report output."""
        report_dict = report.as_dict()
        drift_flags = [
            metric["result"]["drift_detected"]
            for metric in report_dict.get("metrics", [])
            if isinstance(metric.get("result"), dict)
            and "drift_detected" in metric["result"]
        ]
        return sum(drift_flags) / len(drift_flags) if drift_flags else 0.0

    def generate_monitoring_report(
        self,
        model_name: str,
        trained_dataset: str,
        new_dataset: str,
        drift_fail_threshold: float | None = 0.5,
    ) -> dict[str, float | str]:
        fail_threshold = 0.5 if drift_fail_threshold is None else drift_fail_threshold
        model = load_model(self.model_dir / model_name)

        reference_df = self.load_model_ready_dataset(trained_dataset)
        current_df = self.load_model_ready_dataset(new_dataset)

        X_reference, y_reference = self.split_dataset(reference_df)
        X_current, y_current = self.split_dataset(current_df)
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

        numerical_features = X_reference.select_dtypes(
            include=["number"]
        ).columns.tolist()
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

        reports_dir = ensure_directory(self.reports_dir / "drift_monitoring")
        ref_version = Path(trained_dataset).stem
        current_version = Path(new_dataset).stem
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        report_path = (
            reports_dir
            / f"drift_report_{current_version}_vs_{ref_version}_{timestamp}.html"
        )
        report.save_html(report_path.as_posix())

        score = self._drift_score(report)
        status = self.drift_status(score, fail_threshold)

        logger.info("Drift monitoring status: %s", status)
        logger.info("Drift score: %.2f", score)
        logger.info("Saved drift report to %s", report_path.resolve())

        return {
            "status": status,
            "drift_score": score,
            "report_path": str(report_path.resolve()),
        }


def generate_monitoring_report(
    model_name: str,
    trained_dataset: str,
    new_dataset: str,
    drift_fail_threshold: float | None = 0.5,
) -> dict[str, float | str]:
    return DriftMonitor().generate_monitoring_report(
        model_name=model_name,
        trained_dataset=trained_dataset,
        new_dataset=new_dataset,
        drift_fail_threshold=drift_fail_threshold,
    )
