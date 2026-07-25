import pandas as pd

from fraud_detection.data_ingestion import generate_transactions_data
from fraud_detection.feature_engineering import engineer_transaction_features
from fraud_detection.feature_schema import FEATURE_COLUMNS, TARGET_COLUMN


EXPECTED_RAW_COLUMNS = {
    "tx_id",
    "timestamp",
    "user_id",
    "amount",
    "category",
    "merchant_name",
    "merchant_city",
    "merchant_country",
    "channel",
    "device_id",
    "auth_method",
    "lat",
    "lon",
    "ip_address",
    "customer_segment",
    "transaction_status",
    "fraud_pattern",
    "is_fraud",
}


def test_generated_transactions_include_realistic_card_fields(tmp_path):
    transactions = generate_transactions_data(
        n_tx=2_500,
        n_users=120,
        seed=2026,
        output_dir=tmp_path,
    )

    assert EXPECTED_RAW_COLUMNS.issubset(transactions.columns)
    assert transactions["amount"].gt(0).all()
    assert transactions["customer_segment"].nunique() >= 3
    assert transactions["channel"].nunique() == 3
    assert set(transactions["transaction_status"]).issubset({"approved", "declined"})

    fraud_rate = transactions["is_fraud"].mean()
    assert 0.005 <= fraud_rate <= 0.035

    fraud = transactions[transactions["is_fraud"].eq(1)]
    assert fraud["fraud_pattern"].nunique() >= 3
    assert set(fraud["fraud_pattern"]).issubset(
        {"high_amount", "stolen_card", "impossible_travel", "card_testing"}
    )


def test_generated_transactions_are_feature_engineering_compatible(tmp_path):
    transactions = generate_transactions_data(
        n_tx=600,
        n_users=40,
        seed=77,
        output_dir=tmp_path,
    )

    features = engineer_transaction_features(transactions)

    assert list(features.columns) == FEATURE_COLUMNS + [TARGET_COLUMN]
    assert features.shape[0] == transactions.shape[0]
    assert features[FEATURE_COLUMNS].isna().sum().sum() == 0


def test_generation_is_reproducible_for_seed(tmp_path):
    first = generate_transactions_data(
        n_tx=200,
        n_users=30,
        seed=55,
        output_dir=tmp_path / "first",
    )
    second = generate_transactions_data(
        n_tx=200,
        n_users=30,
        seed=55,
        output_dir=tmp_path / "second",
    )

    pd.testing.assert_frame_equal(first, second)
