from pathlib import Path

import pandas as pd
import pytest

from fraud_detection.feature_schema import FEATURE_COLUMNS
from fraud_detection.model_io import load_model, predict_fraud_probability

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = PROJECT_ROOT / "artifacts" / "models" / "xgboost_seed_42.json"


@pytest.fixture(scope="module")
def model():
    return load_model(MODEL_PATH)


def transaction_features(amount: float) -> pd.DataFrame:
    """Create one model-ready transaction row for tests."""
    row = dict.fromkeys(FEATURE_COLUMNS, 0.0)
    row.update(
        {
            "amount": amount,
            "lat": 12.9716,
            "lon": 77.5946,
            "hour": 14,
            "day_of_week": 2,
            "tx_count_24h": 3,
            "avg_spend_user": 350.0,
            "amount_ratio": amount / 350.0,
            "dist_from_last_tx_km": 1.2,
            "travel_velocity_kmph": 5.0,
            "auth_method_PIN": 1.0,
            "category_food": 1.0,
        }
    )
    return pd.DataFrame([row], columns=FEATURE_COLUMNS).astype("float32")


def test_model_output_range(model):
    probability = predict_fraud_probability(model, transaction_features(amount=20.0))[0]
    assert 0 <= probability <= 1


def test_high_amount_risk_does_not_drop_sharply(model):
    low_risk = predict_fraud_probability(model, transaction_features(amount=20.0))[0]
    high_risk = predict_fraud_probability(model, transaction_features(amount=10_000.0))[0]

    assert high_risk >= low_risk * 0.7


def test_feature_schema_has_expected_model_width():
    assert len(FEATURE_COLUMNS) == 17
