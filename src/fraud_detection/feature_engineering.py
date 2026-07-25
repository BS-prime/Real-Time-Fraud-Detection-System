"""Feature engineering for fraud model training."""

from pathlib import Path
import logging
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


class FeatureEngineer:
    """Convert raw transactions into model-ready features."""

    def __init__(
        self,
        feature_columns: Sequence[str] = FEATURE_COLUMNS,
        target_column: str = TARGET_COLUMN,
        simulated_dir: Path = SIMULATED_DATA_DIR,
        feature_dir: Path = FEATURE_DATA_DIR,
    ) -> None:
        self.feature_columns = list(feature_columns)
        self.target_column = target_column
        self.simulated_dir = simulated_dir
        self.feature_dir = feature_dir

    def load_transactions(self, csv_name: str) -> pd.DataFrame:
        csv_path = self.simulated_dir / csv_name
        if not csv_path.exists():
            raise FileNotFoundError(f"Transaction file not found: {csv_path}")
        return pd.read_csv(csv_path)

    def engineer_transaction_features(self, transactions: pd.DataFrame) -> pd.DataFrame:
        df = transactions.copy()
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        df = df.sort_values(["user_id", "timestamp"]).reset_index(drop=True)

        df["hour"] = df["timestamp"].dt.hour
        df["day_of_week"] = df["timestamp"].dt.dayofweek
        df["tx_count_24h"] = (
            df.groupby("user_id").rolling("24h", on="timestamp")["tx_id"].count().values
        )

        df["avg_spend_user"] = df.groupby("user_id")["amount"].transform(
            lambda spend: spend.shift(1).expanding().mean()
        )
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

        df = df.drop(columns=NON_MODEL_COLUMNS)

        for column in self.feature_columns:
            if column not in df.columns:
                df[column] = 0

        numeric_columns = df.select_dtypes(include="number").columns
        df[numeric_columns] = df[numeric_columns].fillna(0)

        return df[self.feature_columns + [self.target_column]]

    def save_features(self, features: pd.DataFrame, seed: int) -> Path:
        output_dir = ensure_directory(self.feature_dir)
        output_path = output_dir / f"fraud_features_seed_{seed}.csv"
        features.to_csv(output_path, index=False)
        logger.info(
            "Saved engineered features to %s with shape %s",
            output_path,
            features.shape,
        )
        return output_path

    def feature_engineer(
        self, csv_name: str = "simulated_transactions_seed_42.csv"
    ) -> pd.DataFrame:
        seed = seed_from_filename(csv_name)
        transactions = self.load_transactions(csv_name)
        features = self.engineer_transaction_features(transactions)
        self.save_features(features, seed)
        return features


def load_transactions(csv_name: str) -> pd.DataFrame:
    return FeatureEngineer().load_transactions(csv_name)


def engineer_transaction_features(transactions: pd.DataFrame) -> pd.DataFrame:
    return FeatureEngineer().engineer_transaction_features(transactions)


def feature_engineer(
    csv_name: str = "simulated_transactions_seed_42.csv",
) -> pd.DataFrame:
    return FeatureEngineer().feature_engineer(csv_name)
