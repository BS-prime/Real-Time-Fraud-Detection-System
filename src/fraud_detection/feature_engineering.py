"""Feature engineering for fraud model training."""

import pandas as pd
import numpy as np

from fraud_detection.feature_schema import (
    AUTH_METHODS,
    CATEGORIES,
    FEATURE_COLUMNS,
    TARGET_COLUMN,
)
from fraud_detection.geo import haversine_km
from fraud_detection.naming import seed_from_filename
from fraud_detection.paths import FEATURE_DATA_DIR, SIMULATED_DATA_DIR, ensure_directory

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


def load_transactions(csv_name: str) -> pd.DataFrame:
    """Load a raw synthetic transaction CSV from ``data/simulated``."""
    csv_path = SIMULATED_DATA_DIR / csv_name
    if not csv_path.exists():
        raise FileNotFoundError(f"Transaction file not found: {csv_path}")
    return pd.read_csv(csv_path)


def engineer_transaction_features(transactions: pd.DataFrame) -> pd.DataFrame:
    """Convert raw transactions into model-ready numeric features."""
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
        (df["timestamp"] - df["prev_ts"]).dt.total_seconds().div(3600).clip(lower=1e-3)
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

    for column in FEATURE_COLUMNS:
        if column not in df.columns:
            df[column] = 0

    numeric_columns = df.select_dtypes(include="number").columns
    df[numeric_columns] = df[numeric_columns].fillna(0)

    return df[FEATURE_COLUMNS + [TARGET_COLUMN]]


def feature_engineer(
    csv_name: str = "simulated_transactions_seed_42.csv",
) -> pd.DataFrame:
    """Engineer raw transactions and save model-ready features."""
    seed = seed_from_filename(csv_name)
    transactions = load_transactions(csv_name)
    features = engineer_transaction_features(transactions)

    output_dir = ensure_directory(FEATURE_DATA_DIR)
    output_path = output_dir / f"fraud_features_seed_{seed}.csv"
    features.to_csv(output_path, index=False)

    print("=" * 70)
    print(f"Saved features  : {output_path.name}")
    print(f"Output directory: {output_dir}")
    print(f"Rows            : {features.shape[0]}")
    print(f"Columns         : {features.shape[1]}")
    print("=" * 70)

    return features
