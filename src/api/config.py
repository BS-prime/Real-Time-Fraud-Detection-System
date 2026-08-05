"""Configuration values for the FastAPI application."""

from dataclasses import dataclass, field
from pathlib import Path
from types import MappingProxyType
from typing import Mapping

from fraud_detection.paths import MODEL_DIR, THRESHOLD_DIR


@dataclass(frozen=True)
class UserHistory:
    """Minimal user history needed for real-time feature creation."""

    avg_spend: float
    last_lat: float
    last_lon: float


DEFAULT_USER_HISTORY = MappingProxyType(
    {
        "user_123": UserHistory(
            avg_spend=45.0,
            last_lat=34.05,
            last_lon=-118.24,
        )
    }
)


@dataclass(frozen=True)
class ApiSettings:
    """Runtime settings for model serving."""

    app_title: str = "Fraud Guard 2026"
    app_version: str = "0.1.0"
    model_version: str = "XGBoost_v:1.0"
    model_path: Path = MODEL_DIR / "xgboost_seed_42.json"
    threshold_path: Path = THRESHOLD_DIR / "optimal_threshold_xgboost_seed_42.json"
    global_avg_spend: float = 60.0
    default_time_delta_min: float = 6.0
    default_tx_count_24h: int = 3
    default_lat: float = 0.0
    default_lon: float = 0.0
    user_history: Mapping[str, UserHistory] = field(
        default_factory=lambda: DEFAULT_USER_HISTORY
    )

    def __post_init__(self) -> None:
        """Validate or normalize runtime settings after initialization."""
        if self.default_time_delta_min <= 0:
            raise ValueError("default_time_delta_min must be positive")
