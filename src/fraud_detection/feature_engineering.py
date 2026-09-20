"""
Feature engineering for fraud model training.

This module converts raw transaction logs into a model-ready feature
matrix. It is used both for training (where the target label is
present) and for batch/online inference (where it is not), so the
target column is treated as optional throughout.
"""

from __future__ import annotations

import logging
import os
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml
from pandas import DataFrame

from fraud_detection.feature_schema import (
    AUTH_METHODS,
    CATEGORIES,
    FEATURE_COLUMNS,
    TARGET_COLUMN,
)
from fraud_detection.geo import haversine_km
from fraud_detection.paths import (
    PARAMS_CONFIG_PATH,
    create_dir,
    simulated_transactions_path,
    test_feature_file_path,
    train_feature_file_path,
)

logger = logging.getLogger(__name__)

__all__ = [
    "FeatureEngineer",
    "FeatureEngineeringError",
    "SchemaValidationError",
    "batch_feature_engineer",
    "engineer_transaction_features",
    "load_transactions",
]

# Columns that must exist in the raw transaction Parquet for feature
# engineering to run at all. Distinct from NON_MODEL_COLUMNS below,
# which is about what survives into the final model matrix.
REQUIRED_RAW_COLUMNS = [
    "tx_id",
    "user_id",
    "timestamp",
    "amount",
    "lat",
    "lon",
    "auth_method",
    "category",
]

# Columns that are used in feature engineering but are not part of the
# final model matrix. These are dropped after feature engineering is complete, so that the final feature matrix contains only model-ready features and the target column (if present).
NON_MODEL_COLUMNS = [
    "tx_id",
    "prev_lat",
    "prev_lon",
    "prev_ts",
    "timestamp",
    "user_id",
    "device_id",
    "ip_address",
]


class FeatureEngineeringError(Exception):
    """
    Base class for feature-engineering failures.
    """


class SchemaValidationError(FeatureEngineeringError):
    """
    Raised when the input transaction data doesn't match expectations.
    """


class FeatureEngineer:
    """
    Convert raw transactions into model-ready features.
    """

    def __init__(
            self,
            params_config_path: Path = PARAMS_CONFIG_PATH,
            transactions_path: Path | None = None,
            train_output_path: Path | None = None,
            test_output_path: Path | None = None,
    ) -> None:
        self.params_config_path = params_config_path
        self.params_config = self._load_params_config()
        self.seed = self.params_config.get("seed", 42)
        self.transactions_path = transactions_path or simulated_transactions_path(self.seed)
        self.train_output_path = train_output_path or train_feature_file_path(self.seed)
        self.test_output_path = test_output_path or test_feature_file_path(self.seed)

    def _load_params_config(self) -> dict[str, Any]:
        """
        Read the model training parameters from YAML configuration.
        """

        if not self.params_config_path.exists():
            raise FileNotFoundError(
                f"Training parameters configuration not found: {self.params_config_path}"
            )

        with self.params_config_path.open("r", encoding="utf-8") as file:
            return yaml.safe_load(file)["feature_engineering"]

    def load_transactions(self, transactions_path: Path | None = None) -> pd.DataFrame:
        """
        Load a raw transaction Parquet from the simulated data directory and validate that it contains the required columns. Returns a DataFrame of the raw transactions.
        """

        transactions_path = transactions_path or self.transactions_path
        if not transactions_path.exists():
            raise FileNotFoundError(f"Transaction file not found: {transactions_path}")

        logger.info("Loading parquet from: %s", transactions_path)
        return pd.read_parquet(transactions_path)

    def train_test_split(
            self, features: pd.DataFrame
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """
        Split the feature DataFrame into training and testing sets based on the configured test size.
        """

        test_size = self.params_config.get("test_size", 0.2)

        if not (0 < test_size < 1):
            raise ValueError(
                f"Invalid test_size: {test_size}. Must be between 0 and 1."
            )

        index = len(features) * test_size

        # Split based on time ordering to avoid data leakage from future transactions into the training set.
        features = features.sort_values(["timestamp", "user_id"]).reset_index(drop=True)
        train_df = features[: int(index)]
        test_df = features[int(index):]

        return train_df, test_df

    @staticmethod
    def save_features(features: pd.DataFrame, file_path: Path) -> None:
        """
        Save the engineered feature matrix to Parquet in the feature data directory.
        """

        output_dir = create_dir(file_path.parent)
        output_path = output_dir / file_path.name

        fd, tmp_name = tempfile.mkstemp(
            prefix=output_path.stem + ".", suffix=".tmp.parquet", dir=output_dir
        )
        os.close(fd)
        tmp_path = Path(tmp_name)
        try:
            features.to_parquet(tmp_path, index=False)
            os.replace(tmp_path, output_path)
        except Exception:
            tmp_path.unlink(missing_ok=True)
            raise

        logger.info(
            "Saved engineered features to %s with shape %s",
            output_path,
            features.shape,
        )

    def engineer_transaction_features(self, transactions: pd.DataFrame) -> pd.DataFrame:
        """
        Engineer features from raw transaction data.
        Returns a DataFrame containing only the model-ready feature columns
        """

        self._validate_raw_schema(transactions)

        df = transactions.copy()
        df = self._prepare_time_ordering(df)
        df = self._add_calendar_features(df)
        df = self._add_spend_features(df)
        df = self._add_geo_velocity_features(df)
        df = self._encode_categorical(df)
        df = df.drop(columns=[c for c in NON_MODEL_COLUMNS if c in df.columns])
        df = self._align_to_feature_schema(df)

        return df

    def _validate_raw_schema(self, transactions: pd.DataFrame) -> None:
        """
        Validate that the input DataFrame has the required columns and no duplicate transaction IDs. Raises SchemaValidationError if validation fails.
        """

        if transactions.empty:
            raise SchemaValidationError("Input transaction DataFrame has no rows")

        missing = [c for c in REQUIRED_RAW_COLUMNS if c not in transactions.columns]
        if missing:
            raise SchemaValidationError(f"Missing required column(s): {missing}")

        if (
                self.params_config.get("require_target")
                and self.params_config.get("target_column") not in transactions.columns
        ):
            raise SchemaValidationError(
                f"Target column '{self.params_config.get('target_column')}' is required but absent. "
                "Pass require_target=False in FeatureEngineeringConfig for inference-time use."
            )

        if transactions["tx_id"].duplicated().any():
            dup_count = int(transactions["tx_id"].duplicated().sum())
            raise SchemaValidationError(
                f"Found {dup_count} duplicate tx_id value(s); each transaction must be unique"
            )

    @staticmethod
    def _prepare_time_ordering(df: pd.DataFrame) -> pd.DataFrame:
        """
        Prepare the DataFrame for time-based operations by ensuring the timestamp column is properly formatted.
        """

        try:
            df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="raise")
        except (ValueError, TypeError) as exc:
            raise SchemaValidationError(
                f"Could not parse 'timestamp' column: {exc}"
            ) from exc

        if df["timestamp"].isna().any():
            raise SchemaValidationError(
                "'timestamp' column contains unparseable values (NaT)"
            )

        return df.sort_values(["user_id", "timestamp"]).reset_index(drop=True)

    def _add_calendar_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add calendar-based features such as hour of day, day of week, and transaction count in the last 24 hours for each user.
        """

        # 1. create new time-series features
        df["hour"] = df["timestamp"].dt.hour
        df["day_of_week"] = df["timestamp"].dt.dayofweek

        # 2. calculate 24-hour rolling window transaction count per user
        df["tx_count_24h"] = (
            df.groupby("user_id")
            .rolling(self.params_config.get("rolling_window", "24h"), on="timestamp")["tx_id"]
            .count()
            .values
        )
        return df

    @staticmethod
    def _add_spend_features(df: pd.DataFrame) -> pd.DataFrame:
        """
        Add features related to transaction amounts, including the average spend of the user and the ratio of the current amount to the average spend.
        """

        if (df["amount"] < 0).any():
            n_negative = int((df["amount"] < 0).sum())
            raise SchemaValidationError(
                f"'amount' contains {n_negative} negative value(s); expected non-negative spend"
            )

        # Expanding mean of *prior* transactions only (shift(1)) to avoid
        # leaking the current transaction's amount into its own feature.
        df["avg_spend_user"] = df.groupby("user_id")["amount"].transform(
            lambda spend: spend.shift(1).expanding().mean()
        )
        # A user's first transaction has no prior history -> avg_spend_user
        # is NaN and the ratio is defined as 0 (neutral, not "spent nothing").
        df["amount_ratio"] = np.where(
            df["avg_spend_user"].gt(0),
            df["amount"] / df["avg_spend_user"],
            0.0,
        )
        return df

    def _add_geo_velocity_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add geographical and velocity-based features.
        """

        df["prev_lat"] = df.groupby("user_id")["lat"].shift(1)
        df["prev_lon"] = df.groupby("user_id")["lon"].shift(1)
        df["prev_ts"] = df.groupby("user_id")["timestamp"].shift(1)

        # Compute distance from the previous transaction in kilometers using the haversine formula. If there is no previous transaction, fill with 0.0.
        df["dist_from_last_tx_km"] = haversine_km(
            df["lat"], df["lon"], df["prev_lat"], df["prev_lon"]
        ).fillna(0.0)

        time_delta = df["timestamp"] - df["prev_ts"]
        if (time_delta.dropna() < pd.Timedelta(0)).any():
            raise SchemaValidationError(
                "Found negative time deltas between consecutive transactions for a "
                "user; timestamps are not monotonically increasing after sort"
            )

        hours_since_previous = (
            time_delta.dt.total_seconds()
            .div(3600)
            .clip(lower=self.params_config["min_hours_between_tx"])
        )

        df["travel_velocity_kmph"] = (
                df["dist_from_last_tx_km"] / hours_since_previous
        ).fillna(0.0)

        return df

    @staticmethod
    def _encode_categorical(df: pd.DataFrame) -> pd.DataFrame:
        """
        Encode categorical features as one-hot (dummy) variables. Warn if any unknown categories are present, and treat them as all-zero dummies.
        """

        for column, known_values in (
                ("auth_method", AUTH_METHODS),
                ("category", CATEGORIES),
        ):
            unknown = set(df[column].dropna().unique()) - set(known_values)
            if unknown:
                logger.warning(
                    "Column '%s' contains %d value(s) outside the known schema: %s. "
                    "These rows will encode to all-zero dummies for this feature.",
                    column,
                    len(unknown),
                    sorted(unknown),
                )
            df[column] = pd.Categorical(df[column], categories=known_values)

        return pd.get_dummies(
            df, columns=["auth_method", "category"], drop_first=True, dtype=int
        )

    @staticmethod
    def _align_to_feature_schema(df: pd.DataFrame) -> pd.DataFrame:
        """
        Ensure that the DataFrame contains all expected feature columns, filling missing ones with zeros. Also fill any remaining Nans in numeric columns with zeros. Finally, return only the columns that are part of the model schema, optionally including the target column.
        """

        for column in FEATURE_COLUMNS:
            if column not in df.columns:
                df[column] = 0

        numeric_columns = df.select_dtypes(include="number").columns
        df[numeric_columns] = df[numeric_columns].fillna(0)

        output_columns = FEATURE_COLUMNS + [TARGET_COLUMN]

        return df[output_columns]

    def feature_engineer(
            self, transactions_path: Path | None = None
    ) -> tuple[Path, Path]:
        """
        Load, engineer, and persist features for a single raw Parquet file.
        """

        transactions = self.load_transactions(transactions_path)
        train_df, test_df = self.train_test_split(transactions)
        train_df = self.engineer_transaction_features(train_df)
        self.save_features(train_df, self.train_output_path)
        self.save_features(test_df, self.test_output_path)
        return self.train_output_path, self.test_output_path


# FEATURE ENGINEERING API FUNCTIONS


def load_transactions(transactions_path: Path) -> DataFrame:
    """
    Validate and load a raw transaction Parquet from the simulated data directory. Returns a DataFrame of the raw transactions.
    """

    return FeatureEngineer().load_transactions(transactions_path)


def engineer_transaction_features(transactions: pd.DataFrame) -> pd.DataFrame:
    """
    Engineer features for a DataFrame of transactions.
    """

    return FeatureEngineer().engineer_transaction_features(transactions)


def batch_feature_engineer(
        transactions_path: Path | None = None,
) -> tuple[Path, Path]:
    """
    Perform feature engineering and save train and test as Parquet.
    """

    return FeatureEngineer().feature_engineer(transactions_path)


if __name__ == "__main__":
    engineer = FeatureEngineer()
    engineer.feature_engineer(transactions_path=engineer.transactions_path)
