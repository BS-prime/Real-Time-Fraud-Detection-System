"""FastAPI dependency providers."""

from functools import lru_cache

from .config import ApiSettings
from .services import FraudScoringService


@lru_cache
def get_settings() -> ApiSettings:
    """Return cached API settings."""
    return ApiSettings()


@lru_cache
def get_scoring_service() -> FraudScoringService:
    """Return the cached scoring service used by prediction routes."""
    return FraudScoringService(get_settings())
