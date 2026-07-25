"""Pydantic request and response contracts for the API."""

from typing import Literal

from pydantic import BaseModel, Field

AuthMethod = Literal["Biometric", "PIN", "Password"]
MerchantCategory = Literal[
    "food",
    "grocery",
    "tech",
    "travel",
    "utilities",
    "entertainment",
]
RiskBand = Literal["LOW", "MEDIUM", "HIGH", "VERY_HIGH"]
RecommendedAction = Literal["ALLOW", "CHALLENGE", "BLOCK"]


class TransactionRequest(BaseModel):
    """Request body for a single transaction scoring call."""

    user_id: str = Field(min_length=1)
    amount: float = Field(gt=0)
    lat: float | None = Field(default=None, ge=-90.0, le=90.0)
    lon: float | None = Field(default=None, ge=-180.0, le=180.0)
    auth_method: AuthMethod
    category: MerchantCategory
    time_delta_min: float | None = Field(default=None, gt=0)
    tx_count_24h: int | None = Field(default=3, ge=0)


class PredictionResponse(BaseModel):
    """Response returned after scoring a transaction."""

    model_version: str
    fraud_probability: float = Field(ge=0, le=1)
    risk_band: RiskBand
    recommended_action: RecommendedAction
    decision_reasons: list[str]
    fallbacks_used: list[str]


class HealthResponse(BaseModel):
    """Basic health-check response."""

    status: str


class ReadinessResponse(BaseModel):
    """Readiness response proving the model and threshold are loaded."""

    status: str
    model_version: str
    threshold: float
