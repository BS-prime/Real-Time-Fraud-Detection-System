from datetime import datetime

from api.config import ApiSettings
from api.schemas import TransactionRequest
from api.services import FraudScoringService
from fraud_detection.feature_schema import FEATURE_COLUMNS


def sample_transaction() -> TransactionRequest:
    return TransactionRequest(
        user_id="user_123",
        amount=2000,
        lat=40.7128,
        lon=-74.0060,
        auth_method="Biometric",
        category="food",
    )


def test_build_model_features_uses_shared_feature_schema():
    service = FraudScoringService(ApiSettings())
    feature_result = service.build_model_features(
        sample_transaction(),
        now=datetime(2026, 7, 10, 14, 30),
    )

    assert feature_result.features.shape == (1, len(FEATURE_COLUMNS))
    assert list(feature_result.features.columns) == FEATURE_COLUMNS
    assert feature_result.fallbacks_used == ["DEFAULT_TIME_DELTA_USED"]


def test_score_returns_prediction_response_contract():
    service = FraudScoringService(ApiSettings())
    response = service.score(sample_transaction(), now=datetime(2026, 7, 10, 14, 30))

    assert response.model_version == "XGBoost_v:1.0"
    assert 0 <= response.fraud_probability <= 1
    assert response.risk_band in {"LOW", "MEDIUM", "HIGH", "VERY_HIGH"}
    assert response.recommended_action in {"ALLOW", "CHALLENGE", "BLOCK"}
