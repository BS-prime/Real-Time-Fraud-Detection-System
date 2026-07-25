"""Application services used by the API routes."""

from dataclasses import dataclass
from datetime import datetime

import pandas as pd

from fraud_detection.feature_schema import FEATURE_COLUMNS
from fraud_detection.geo import haversine_km
from fraud_detection.model_io import load_model, load_threshold, predict_fraud_probability

from .config import ApiSettings, UserHistory
from .schemas import PredictionResponse, RecommendedAction, RiskBand, TransactionRequest


@dataclass(frozen=True)
class FeatureBuildResult:
    """Model features plus values used to explain the decision."""

    features: pd.DataFrame
    amount_ratio: float
    travel_velocity_kmph: float
    tx_count_24h: int
    fallbacks_used: list[str]


class FraudScoringService:
    """Build features, run inference, and translate scores into business decisions."""

    def __init__(self, settings: ApiSettings) -> None:
        self.settings = settings
        self.model = load_model(settings.model_path)
        self.threshold = load_threshold(settings.threshold_path)

    def score(
        self,
        transaction: TransactionRequest,
        now: datetime | None = None,
    ) -> PredictionResponse:
        """Score one transaction and return the public API response."""
        feature_result = self.build_model_features(transaction, now=now)
        probability = float(
            predict_fraud_probability(self.model, feature_result.features)[0]
        )

        return PredictionResponse(
            model_version=self.settings.model_version,
            fraud_probability=round(probability, 4),
            risk_band=self.risk_band(probability),
            recommended_action=self.recommended_action(probability),
            decision_reasons=self.decision_reasons(
                feature_result.amount_ratio,
                feature_result.travel_velocity_kmph,
                feature_result.tx_count_24h,
            ),
            fallbacks_used=feature_result.fallbacks_used,
        )

    def build_model_features(
        self,
        transaction: TransactionRequest,
        now: datetime | None = None,
    ) -> FeatureBuildResult:
        """Create the exact feature table expected by the trained model."""
        now = now or datetime.now()
        history, fallbacks = self.user_history_for(transaction)

        if transaction.time_delta_min is None:
            fallbacks.append("DEFAULT_TIME_DELTA_USED")

        if (
            transaction.lat is None or transaction.lon is None
        ) and "DEFAULT_LOCATION_USED" not in fallbacks:
            fallbacks.append("REQUEST_LOCATION_MISSING")

        lat = transaction.lat if transaction.lat is not None else history.last_lat
        lon = transaction.lon if transaction.lon is not None else history.last_lon
        tx_count_24h = (
            self.settings.default_tx_count_24h
            if transaction.tx_count_24h is None
            else transaction.tx_count_24h
        )
        time_delta_min = (
            self.settings.default_time_delta_min
            if transaction.time_delta_min is None
            else transaction.time_delta_min
        )

        amount_ratio = transaction.amount / max(history.avg_spend, 1e-6)
        distance_km = float(haversine_km(lat, lon, history.last_lat, history.last_lon))
        travel_velocity_kmph = distance_km / (time_delta_min / 60.0)

        row = {
            "amount": transaction.amount,
            "lat": lat,
            "lon": lon,
            "hour": now.hour,
            "day_of_week": now.weekday(),
            "tx_count_24h": tx_count_24h,
            "avg_spend_user": history.avg_spend,
            "amount_ratio": amount_ratio,
            "dist_from_last_tx_km": distance_km,
            "travel_velocity_kmph": travel_velocity_kmph,
            "auth_method_PIN": int(transaction.auth_method == "PIN"),
            "auth_method_Password": int(transaction.auth_method == "Password"),
            "category_food": int(transaction.category == "food"),
            "category_grocery": int(transaction.category == "grocery"),
            "category_tech": int(transaction.category == "tech"),
            "category_travel": int(transaction.category == "travel"),
            "category_utilities": int(transaction.category == "utilities"),
        }

        features = pd.DataFrame([row], columns=FEATURE_COLUMNS).astype("float32")
        return FeatureBuildResult(
            features=features,
            amount_ratio=amount_ratio,
            travel_velocity_kmph=travel_velocity_kmph,
            tx_count_24h=tx_count_24h,
            fallbacks_used=fallbacks,
        )

    def user_history_for(
        self,
        transaction: TransactionRequest,
    ) -> tuple[UserHistory, list[str]]:
        """Return stored user history or transparent defaults."""
        user_key = transaction.user_id.strip().lower()
        if user_key in self.settings.user_history:
            return self.settings.user_history[user_key], []

        fallbacks = ["GLOBAL_AVG_SPEND_USED", "NO_LOCATION_HISTORY"]
        if transaction.lat is None or transaction.lon is None:
            fallbacks.append("DEFAULT_LOCATION_USED")

        return (
            UserHistory(
                avg_spend=self.settings.global_avg_spend,
                last_lat=(
                    transaction.lat
                    if transaction.lat is not None
                    else self.settings.default_lat
                ),
                last_lon=(
                    transaction.lon
                    if transaction.lon is not None
                    else self.settings.default_lon
                ),
            ),
            fallbacks,
        )

    def recommended_action(self, probability: float) -> RecommendedAction:
        """Choose the business action for a scored transaction."""
        if probability >= self.threshold:
            return "BLOCK"
        if probability >= 0.50:
            return "CHALLENGE"
        return "ALLOW"

    @staticmethod
    def risk_band(probability: float) -> RiskBand:
        """Map fraud probability to a human-readable risk band."""
        if probability >= 0.85:
            return "VERY_HIGH"
        if probability >= 0.65:
            return "HIGH"
        if probability >= 0.40:
            return "MEDIUM"
        return "LOW"

    @staticmethod
    def decision_reasons(
        amount_ratio: float,
        travel_velocity_kmph: float,
        tx_count_24h: int,
    ) -> list[str]:
        """Return concise explanation signals for the response."""
        reasons = []

        if amount_ratio >= 3:
            reasons.append(
                "Transaction amount significantly higher than user's normal spending"
            )
        if travel_velocity_kmph >= 800:
            reasons.append("Transaction location implies unrealistic travel speed")
        if tx_count_24h >= 20:
            reasons.append("Unusually high number of transactions in past 24 hours")
        if amount_ratio >= 10 or travel_velocity_kmph >= 900:
            reasons.append("Extreme deviation from normal transaction behavior")

        return reasons[:3]
