"""
Feature engineering for fraud model training.

This module converts raw transaction logs into a model-ready feature
matrix. It is used both for training (where the target label is
present) and for batch/online inference (where it is not), so the
target column is treated as optional throughout.

Design notes
------------
- All transformations are pure functions of the input DataFrame; the
  class holds only configuration, so instances are cheap, stateless,
  and safe to share across threads.
- Every step that can fail on malformed input (missing columns, bad
  dtypes, out-of-order timestamps, unknown categories) fails loudly
  with a specific exception rather than silently producing NaNs or
  zeros, because silent zero-filling of a fraud feature is a
  correctness bug, not a convenience.
- Writing output is atomic (write to a temp file, then rename) so a
  crash mid-write can never leave a corrupt/partial feature file on
  disk for a downstream training job to pick up.
"""

from __future__ import annotations

import logging
import os
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd

from fraud_detection.feature_schema import (
    AUTH_METHODS,
    CATEGORIES,
    FEATURE_COLUMNS,
    TARGET_COLUMN,
)
from fraud_detection.geo import haversine_km
from fraud_detection.naming import seed_from_filename
from fraud_detection.paths import FEATURE_DATA_DIR, SIMULATED_DATA_DIR, ensure_directory

logger = logging.getLogger(__name__)

__all__ = [
    "FeatureEngineeringConfig",
    "FeatureEngineeringError",
    "SchemaValidationError",
    "FeatureEngineer",
    "load_transactions",
    "engineer_transaction_features",
    "feature_engineer",
]

# Columns that must exist in the raw transaction CSV for feature
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

# Identifier / raw columns that are useful for joins, debugging, and
# feature computation, but must never leak into the model matrix.
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
    """Base class for feature-engineering failures."""


class SchemaValidationError(FeatureEngineeringError):
    """Raised when the input transaction data doesn't match expectations."""


@dataclass(frozen=True)
class FeatureEngineeringConfig:
    """
    Tunable parameters for feature engineering.

    Kept separate from FeatureEngineer's constructor args so that all
    "how do we compute features" knobs live in one serializable place
    (useful for logging config alongside a model run, or for hashing
    config into an experiment/feature-store key).
    """

    rolling_window: str = "24h"
    min_hours_between_tx: float = 1e-3  # floor to avoid div-by-zero on velocity
    feature_columns: Sequence[str] = field(
        default_factory=lambda: list(FEATURE_COLUMNS)
    )
    target_column: str = TARGET_COLUMN
    require_target: bool = True  # False for inference-time feature building


class FeatureEngineer:
    """
    Convert raw transactions into model-ready features.
    """

    def __init__(
        self,
        config: FeatureEngineeringConfig | None = None,
        simulated_dir: Path = SIMULATED_DATA_DIR,
        feature_dir: Path = FEATURE_DATA_DIR,
    ) -> None:
        self.config = config or FeatureEngineeringConfig()
        self.simulated_dir = Path(simulated_dir)
        self.feature_dir = Path(feature_dir)

    # ------------------------------------------------------------------
    # I/O
    # ------------------------------------------------------------------

    def load_transactions(self, csv_name: str) -> pd.DataFrame:
        """Load a raw transaction CSV from the simulated data directory.

        Raises:
            FileNotFoundError: if the CSV doesn't exist.
            SchemaValidationError: if the file is empty or missing
                required columns.
        """

        csv_path = self.simulated_dir / csv_name
        if not csv_path.exists():
            raise FileNotFoundError(f"Transaction file not found: {csv_path}")

        df = pd.read_csv(csv_path)

        if df.empty:
            raise SchemaValidationError(f"{csv_path} contains no rows")

        missing = [c for c in REQUIRED_RAW_COLUMNS if c not in df.columns]
        if missing:
            raise SchemaValidationError(
                f"{csv_path} is missing required column(s): {missing}"
            )

        return df

    def engineer_transaction_features(self, transactions: pd.DataFrame) -> pd.DataFrame:
        """Transform raw transactions into model-ready feature columns."""
        df = transactions.copy()
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        df = df.sort_values(["user_id", "timestamp"]).reset_index(drop=True)

        df["hour"] = df["timestamp"].dt.hour
        df["day_of_week"] = df["timestamp"].dt.dayofweek
        df["tx_count_24h"] = (
            df.groupby("user_id").rolling("24h", on="timestamp")["tx_id"].count().values
        )
        # tx_count_24h counts transactions per user inside a 24-hour window.

        df["avg_spend_user"] = df.groupby("user_id")["amount"].transform(
            lambda spend: spend.shift(1).expanding().mean()
        )
        # amount_ratio compares current transaction amount to the customer's historical average.
        df["amount_ratio"] = np.where(
            df["avg_spend_user"] > 0,
            df["amount"] / df["avg_spend_user"],
            0,
        )

        df["prev_lat"] = df.groupby("user_id")["lat"].shift(1)
        df["prev_lon"] = df.groupby("user_id")["lon"].shift(1)
        df["prev_ts"] = df.groupby("user_id")["timestamp"].shift(1)

        df["dist_from_last_tx_km"] = haversine_km(
            df["lat"],
            df["lon"],
            df["prev_lat"],
            df["prev_lon"],
        ).fillna(0)
        # Fill missing distance values for first transactions where there is no previous location.

        hours_since_previous = (
            (df["timestamp"] - df["prev_ts"]) 
            .dt.total_seconds()
            .div(3600)
            .clip(lower=1e-3)
        )
        df["travel_velocity_kmph"] = df["dist_from_last_tx_km"] / hours_since_previous

        df["auth_method"] = pd.Categorical(df["auth_method"], categories=AUTH_METHODS)
        df["category"] = pd.Categorical(df["category"], categories=CATEGORIES)
        df = pd.get_dummies(
            df,
            columns=["auth_method", "category"],
            drop_first=True,
            dtype=int,
        )
        # Convert categorical features into the binary columns expected by the model.

        df = df.drop(columns=NON_MODEL_COLUMNS)

        for column in self.feature_columns:
            if column not in df.columns:
                df[column] = 0

        numeric_columns = df.select_dtypes(include="number").columns
        df[numeric_columns] = df[numeric_columns].fillna(0)

        return df[self.feature_columns + [self.target_column]]


    def save_features(self, features: pd.DataFrame, seed: int) -> Path:
        """
        Persist engineered features to disk atomically.

        Writes to a temp file in the same directory and renames it into
        place, so a process crash or concurrent read never observes a
        partially written feature file.
        """
        output_dir = ensure_directory(self.feature_dir)
        output_path = output_dir / f"fraud_features_seed_{seed}.csv"

        fd, tmp_name = tempfile.mkstemp(
            prefix=output_path.stem + ".", suffix=".tmp.csv", dir=output_dir
        )
        os.close(fd)
        tmp_path = Path(tmp_name)
        try:
            features.to_csv(tmp_path, index=False)
            os.replace(tmp_path, output_path)  # atomic on POSIX and Windows
        except Exception:
            tmp_path.unlink(missing_ok=True)
            raise

        logger.info(
            "Saved engineered features to %s with shape %s",
            output_path,
            features.shape,
        )
        return output_path

    # ------------------------------------------------------------------
    # Feature computation
    # ------------------------------------------------------------------

    def engineer_transaction_features(self, transactions: pd.DataFrame) -> pd.DataFrame:
        """Compute the full feature matrix from raw transactions.

        Raises:
            SchemaValidationError: if required columns are missing, the
                target column is missing while required, or timestamps
                cannot be parsed.
        """
        self._validate_raw_schema(transactions)

        df = transactions.copy()
        df = self._prepare_time_ordering(df)
        df = self._add_calendar_features(df)
        df = self._add_spend_features(df)
        df = self._add_geo_velocity_features(df)
        df = self._encode_categoricals(df)
        df = df.drop(columns=[c for c in NON_MODEL_COLUMNS if c in df.columns])
        df = self._align_to_feature_schema(df)

        return df

    def _validate_raw_schema(self, transactions: pd.DataFrame) -> None:
        if transactions.empty:
            raise SchemaValidationError("Input transaction DataFrame has no rows")

        missing = [c for c in REQUIRED_RAW_COLUMNS if c not in transactions.columns]
        if missing:
            raise SchemaValidationError(f"Missing required column(s): {missing}")

        if (
            self.config.require_target
            and self.config.target_column not in transactions.columns
        ):
            raise SchemaValidationError(
                f"Target column '{self.config.target_column}' is required but absent. "
                "Pass require_target=False in FeatureEngineeringConfig for inference-time use."
            )

        if transactions["tx_id"].duplicated().any():
            dup_count = int(transactions["tx_id"].duplicated().sum())
            raise SchemaValidationError(
                f"Found {dup_count} duplicate tx_id value(s); each transaction must be unique"
            )

    def _prepare_time_ordering(self, df: pd.DataFrame) -> pd.DataFrame:
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
        df["hour"] = df["timestamp"].dt.hour
        df["day_of_week"] = df["timestamp"].dt.dayofweek
        df["tx_count_24h"] = (
            df.groupby("user_id")
            .rolling(self.config.rolling_window, on="timestamp")["tx_id"]
            .count()
            .values
        )
        return df

    def _add_spend_features(self, df: pd.DataFrame) -> pd.DataFrame:
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
        df["prev_lat"] = df.groupby("user_id")["lat"].shift(1)
        df["prev_lon"] = df.groupby("user_id")["lon"].shift(1)
        df["prev_ts"] = df.groupby("user_id")["timestamp"].shift(1)

        # NaN (no previous tx for this user) -> distance of 0, i.e. "no
        # movement to measure yet", not "traveled zero km".
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
            .clip(lower=self.config.min_hours_between_tx)
        )
        df["travel_velocity_kmph"] = (
            df["dist_from_last_tx_km"] / hours_since_previous
        ).fillna(0.0)
        
        return df

    def _encode_categoricals(self, df: pd.DataFrame) -> pd.DataFrame:
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

    def _align_to_feature_schema(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Ensure every expected feature column exists, then select the
        final column set in a fixed order.

        Unseen dummy columns (e.g. a category value that never appeared
        in this batch) are added as all-zero; this is safe *only* because
        `_encode_categoricals` fixes the categorical levels up front, so
        "missing column" here means "value didn't occur in this batch",
        never "unknown category silently dropped".
        """
        for column in self.config.feature_columns:
            if column not in df.columns:
                df[column] = 0

        numeric_columns = df.select_dtypes(include="number").columns
        df[numeric_columns] = df[numeric_columns].fillna(0)

        output_columns = list(self.config.feature_columns)
        if self.config.require_target:
            output_columns = output_columns + [self.config.target_column]

        return df[output_columns]

    # ------------------------------------------------------------------
    # Convenience orchestration
    # ------------------------------------------------------------------

    def feature_engineer(
        self, csv_name: str = "simulated_transactions_seed_42.csv"
    ) -> pd.DataFrame:
        """
        Load, engineer, and persist features for a single raw CSV.
        """
        seed = seed_from_filename(csv_name)
        transactions = self.load_transactions(csv_name)
        features = self.engineer_transaction_features(transactions)
        self.save_features(features, seed)
        return features


# ----------------------------------------------------------------------
# Module-level convenience wrappers (kept for backward compatibility with
# existing call sites / notebooks). Prefer instantiating FeatureEngineer
# directly in new code so config is explicit and reusable.
# ----------------------------------------------------------------------


def load_transactions(csv_name: str) -> pd.DataFrame:
    return FeatureEngineer().load_transactions(csv_name)


def engineer_transaction_features(transactions: pd.DataFrame) -> pd.DataFrame:
    return FeatureEngineer().engineer_transaction_features(transactions)


def feature_engineer(
    csv_name: str = "simulated_transactions_seed_42.csv",
) -> pd.DataFrame:
    return FeatureEngineer().feature_engineer(csv_name)
