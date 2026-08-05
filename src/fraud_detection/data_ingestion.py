"""Synthetic bank/card transaction data generation."""

from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
import logging
import random

import pandas as pd
from faker import Faker

from fraud_detection.feature_schema import AUTH_METHODS
from fraud_detection.paths import SIMULATED_DATA_DIR, ensure_directory

logger = logging.getLogger(__name__)

START_DATE = datetime(2026, 1, 1, 8, 0)
DEFAULT_FRAUD_RATE = 0.015

CITIES = [
    {"name": "New York", "country": "US", "lat": 40.7128, "lon": -74.0060},
    {"name": "Chicago", "country": "US", "lat": 41.8781, "lon": -87.6298},
    {"name": "Dallas", "country": "US", "lat": 32.7767, "lon": -96.7970},
    {"name": "London", "country": "GB", "lat": 51.5072, "lon": -0.1276},
    {"name": "Bengaluru", "country": "IN", "lat": 12.9716, "lon": 77.5946},
    {"name": "Mumbai", "country": "IN", "lat": 19.0760, "lon": 72.8777},
]

CUSTOMER_SEGMENTS = {
    "student": {"avg_spend": 35, "gap_minutes": 900},
    "everyday": {"avg_spend": 85, "gap_minutes": 600},
    "premium": {"avg_spend": 220, "gap_minutes": 420},
    "business": {"avg_spend": 320, "gap_minutes": 360},
}

CATEGORY_RULES = {
    "grocery": {"typical_amount": 45, "merchant": "FreshBasket"},
    "food": {"typical_amount": 28, "merchant": "Urban Spoon"},
    "entertainment": {"typical_amount": 70, "merchant": "StreamHub"},
    "utilities": {"typical_amount": 140, "merchant": "City Power"},
    "tech": {"typical_amount": 450, "merchant": "GadgetPro"},
    "travel": {"typical_amount": 750, "merchant": "SkyWays"},
}

SEGMENT_CATEGORY_WEIGHTS = {
    "student": [0.27, 0.32, 0.20, 0.06, 0.12, 0.03],
    "everyday": [0.30, 0.22, 0.10, 0.18, 0.12, 0.08],
    "premium": [0.16, 0.18, 0.14, 0.10, 0.20, 0.22],
    "business": [0.10, 0.22, 0.08, 0.08, 0.20, 0.32],
}

CHANNELS = ["pos", "ecommerce", "mobile_wallet"]
FRAUD_PATTERNS = ["high_amount", "stolen_card", "impossible_travel", "card_testing"]


@dataclass
class UserProfile:
    """Basic customer behavior used to make transactions feel personal."""

    user_id: str
    segment: str
    home_city: dict[str, str | float]
    devices: list[str]
    avg_spend: float
    gap_minutes: int
    last_tx_time: datetime


class TransactionSimulator:
    """Generate realistic synthetic transactions for model training."""

    def __init__(
        self,
        n_users: int = 500,
        seed: int = 42,
        fraud_rate: float = DEFAULT_FRAUD_RATE,
        output_dir: Path | str | None = None,
    ) -> None:
        self.n_users = n_users
        self.seed = seed
        self.fraud_rate = fraud_rate
        self.output_dir = (
            Path(output_dir) if output_dir is not None else SIMULATED_DATA_DIR
        )
        self.fake = Faker()
        self.fake.seed_instance(seed)
        self.rng = random.Random(seed)
        self.profiles = self._create_user_profiles()

    def _create_user_profiles(self) -> dict[str, UserProfile]:
        """Build a synthetic user population with spending habits and home locations."""
        profiles: dict[str, UserProfile] = {}

        for user_number in range(self.n_users):
            segment = self.rng.choice(list(CUSTOMER_SEGMENTS))
            segment_rules = CUSTOMER_SEGMENTS[segment]
            user_id = f"user_{user_number}"

            profiles[user_id] = UserProfile(
                user_id=user_id,
                segment=segment,
                home_city=self.rng.choice(CITIES),
                devices=[self.fake.uuid4()[:10] for _ in range(self.rng.randint(1, 3))],
                avg_spend=segment_rules["avg_spend"],
                gap_minutes=segment_rules["gap_minutes"],
                last_tx_time=START_DATE + timedelta(days=self.rng.randint(0, 14)),
            )

        return profiles

    def _generate_normal_transaction(self, profile: UserProfile) -> dict[str, object]:
        """Generate a valid transaction for a user profile before fraud injection."""
        categories = list(CATEGORY_RULES)
        category = self.rng.choices(
            categories,
            weights=SEGMENT_CATEGORY_WEIGHTS[profile.segment],
            k=1,
        )[0]
        category_rules = CATEGORY_RULES[category]

        minutes_since_last_tx = self.rng.randint(5, profile.gap_minutes * 2)
        timestamp = profile.last_tx_time + timedelta(minutes=minutes_since_last_tx)
        profile.last_tx_time = timestamp

        amount = round(self.rng.uniform(0.5, 1.8) * category_rules["typical_amount"], 2)
        if category in {"tech", "travel"}:
            amount = round(amount * self.rng.uniform(1.2, 2.4), 2)

        city = profile.home_city
        if category == "travel":
            city = self.rng.choice(CITIES)

        auth_method = self.rng.choice(AUTH_METHODS)
        lat = round(city["lat"] + self.rng.uniform(-0.05, 0.05), 6)
        lon = round(city["lon"] + self.rng.uniform(-0.05, 0.05), 6)

        return {
            "tx_id": self.fake.uuid4()[:12],
            "timestamp": timestamp,
            "user_id": profile.user_id,
            "amount": amount,
            "category": category,
            "merchant_name": category_rules["merchant"],
            "merchant_city": city["name"],
            "merchant_country": city["country"],
            "channel": self.rng.choice(CHANNELS),
            "device_id": self.rng.choice(profile.devices),
            "auth_method": auth_method,
            "lat": lat,
            "lon": lon,
            "ip_address": self.fake.ipv4(),
            "customer_segment": profile.segment,
            "transaction_status": "approved",
            "fraud_pattern": "legitimate",
            "is_fraud": 0,
        }

    def _apply_fraud_pattern(
        self, transaction: dict[str, object], profile: UserProfile
    ) -> dict[str, object]:
        """Inject a fraud pattern into the transaction based on the configured fraud rate."""
        if self.rng.random() >= self.fraud_rate:
            return transaction

        fraud_pattern = self.rng.choice(FRAUD_PATTERNS)
        transaction["is_fraud"] = 1
        transaction["fraud_pattern"] = fraud_pattern
        transaction["auth_method"] = "Password"

        if fraud_pattern == "high_amount":
            transaction["category"] = "tech"
            transaction["merchant_name"] = CATEGORY_RULES["tech"]["merchant"]
            transaction["amount"] = round(self.rng.uniform(2_000, 4_500), 2)

        elif fraud_pattern == "stolen_card":
            foreign_city = self.rng.choice(
                [city for city in CITIES if city != profile.home_city]
            )
            transaction["device_id"] = f"new_{self.fake.uuid4()[:10]}"
            transaction["merchant_city"] = foreign_city["name"]
            transaction["merchant_country"] = foreign_city["country"]
            transaction["lat"] = foreign_city["lat"]
            transaction["lon"] = foreign_city["lon"]
            transaction["amount"] = round(self.rng.uniform(500, 4_500), 2)

        elif fraud_pattern == "impossible_travel":
            distant_city = self.rng.choice(
                [city for city in CITIES if city != profile.home_city]
            )
            transaction["timestamp"] = profile.last_tx_time + timedelta(minutes=10)
            transaction["merchant_city"] = distant_city["name"]
            transaction["merchant_country"] = distant_city["country"]
            transaction["lat"] = distant_city["lat"]
            transaction["lon"] = distant_city["lon"]
            transaction["amount"] = round(self.rng.uniform(300, 3_000), 2)

        else:
            transaction["category"] = "entertainment"
            transaction["merchant_name"] = CATEGORY_RULES["entertainment"]["merchant"]
            transaction["channel"] = "ecommerce"
            transaction["device_id"] = f"new_{self.fake.uuid4()[:10]}"
            transaction["amount"] = round(self.rng.uniform(1, 15), 2)
            transaction["transaction_status"] = self.rng.choice(
                ["approved", "declined"]
            )

        return transaction

    def _output_path(self) -> Path:
        directory = ensure_directory(self.output_dir)
        return directory / f"simulated_transactions_seed_{self.seed}.csv"

    def generate(self, n_tx: int = 10_000) -> pd.DataFrame:
        rows: list[dict[str, object]] = []

        for _ in range(n_tx):
            profile = self.profiles[self.rng.choice(list(self.profiles))]
            transaction = self._generate_normal_transaction(profile)
            transaction = self._apply_fraud_pattern(transaction, profile)
            profile.last_tx_time = transaction["timestamp"]
            rows.append(transaction)

        return pd.DataFrame(rows)

    def save(self, transactions: pd.DataFrame) -> Path:
        output_path = self._output_path()
        transactions.to_csv(output_path, index=False)
        logger.info(
            "Saved simulated transaction data to %s with %d rows",
            output_path,
            len(transactions),
        )
        return output_path

    def generate_transactions_data(
        self,
        n_tx: int = 10_000,
    ) -> pd.DataFrame:
        transactions = self.generate(n_tx)
        self.save(transactions)
        return transactions


def generate_transactions_data(
    n_tx: int = 10_000,
    n_users: int = 500,
    seed: int = 42,
    output_dir: str | Path | None = None,
) -> pd.DataFrame:
    """Generate synthetic transactions and save them under ``data/simulated``."""
    simulator = TransactionSimulator(
        n_users=n_users,
        seed=seed,
        output_dir=output_dir,
    )
    return simulator.generate_transactions_data(n_tx=n_tx)
