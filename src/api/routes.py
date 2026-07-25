"""HTTP routes for the fraud scoring API."""

from fastapi import APIRouter, Depends, HTTPException

from .dependencies import get_scoring_service
from .schemas import (
    HealthResponse,
    PredictionResponse,
    ReadinessResponse,
    TransactionRequest,
)
from .services import FraudScoringService

router = APIRouter()


@router.get("/", response_model=HealthResponse, tags=["health"])
def root_health_check() -> HealthResponse:
    """Basic liveness check kept at the root for portfolio demos."""
    return HealthResponse(status="Fraud Guard 2026 is up and running.")


@router.get("/health/live", response_model=HealthResponse, tags=["health"])
def liveness_check() -> HealthResponse:
    """Liveness check for process-level health."""
    return HealthResponse(status="ok")


@router.get("/health/ready", response_model=ReadinessResponse, tags=["health"])
def readiness_check(
    service: FraudScoringService = Depends(get_scoring_service),
) -> ReadinessResponse:
    """Readiness check that confirms model-serving dependencies are loaded."""
    return ReadinessResponse(
        status="ready",
        model_version=service.settings.model_version,
        threshold=service.threshold,
    )


@router.post("/predict", response_model=PredictionResponse, tags=["prediction"])
async def predict_fraud(
    transaction: TransactionRequest,
    service: FraudScoringService = Depends(get_scoring_service),
) -> PredictionResponse:
    """Score one transaction and return a business decision."""
    try:
        return service.score(transaction)
    except Exception as error:
        raise HTTPException(status_code=500, detail=f"Inference error: {error}") from error
