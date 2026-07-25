"""Synthetic bank/card transaction data generation.

The goal of this script is to create realistic-enough demo data.

1. Create customer profiles.
2. Generate mostly normal transactions.
3. Inject a small number of fraud patterns.
4. Save the CSV for the next pipeline step.
"""

from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
import random

import pandas as pd
from faker import Faker

from fraud_detection.feature_schema import AUTH_METHODS
from fraud_detection.paths import SIMULATED_DATA_DIR, ensure_directory

START_DATE = datetime(2026, 1, 1, 8, 0)
FRAUD_RATE = 0.015

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
    home_city: dict
    devices: list[str]
    avg_spend: float
    gap_minutes: int
    last_tx_time: datetime


def create_user_profiles(
    fake: Faker,
    rng: random.Random,
    n_users: int,
) -> dict[str, UserProfile]:
    """Create customers with a home location, devices, and spending profile."""
    profiles = {}

    for user_number in range(n_users):
        segment = rng.choice(list(CUSTOMER_SEGMENTS))
        segment_rules = CUSTOMER_SEGMENTS[segment]
        user_id = f"user_{user_number}"

        profiles[user_id] = UserProfile(
            user_id=user_id,
            segment=segment,
            home_city=rng.choice(CITIES),
            devices=[fake.uuid4()[:10] for _ in range(rng.randint(1, 3))],
            avg_spend=segment_rules["avg_spend"],
            gap_minutes=segment_rules["gap_minutes"],
            last_tx_time=START_DATE + timedelta(days=rng.randint(0, 14)),
        )

    return profiles


def generate_normal_transaction(
    fake: Faker,
    rng: random.Random,
    profile: UserProfile,
) -> dict:
    """Generate one normal customer transaction."""
    categories = list(CATEGORY_RULES)
    category = rng.choices(
        categories,
        weights=SEGMENT_CATEGORY_WEIGHTS[profile.segment],
        k=1,
    )[0]
    category_rules = CATEGORY_RULES[category]

    minutes_since_last_tx = rng.randint(5, profile.gap_minutes * 2)
    timestamp = profile.last_tx_time + timedelta(minutes=minutes_since_last_tx)
    profile.last_tx_time = timestamp

    amount = round(
        rng.uniform(0.5, 1.8) * category_rules["typical_amount"],
        2,
    )
    if category in {"tech", "travel"}:
        amount = round(amount * rng.uniform(1.2, 2.4), 2)

    city = profile.home_city
    if category == "travel":
        city = rng.choice(CITIES)

    channel = rng.choice(CHANNELS)
    auth_method = rng.choice(AUTH_METHODS)
    lat = round(city["lat"] + rng.uniform(-0.05, 0.05), 6)
    lon = round(city["lon"] + rng.uniform(-0.05, 0.05), 6)

    return {
        "tx_id": fake.uuid4()[:12],
        "timestamp": timestamp,
        "user_id": profile.user_id,
        "amount": amount,
        "category": category,
        "merchant_name": category_rules["merchant"],
        "merchant_city": city["name"],
        "merchant_country": city["country"],
        "channel": channel,
        "device_id": rng.choice(profile.devices),
        "auth_method": auth_method,
        "lat": lat,
        "lon": lon,
        "ip_address": fake.ipv4(),
        "customer_segment": profile.segment,
        "transaction_status": "approved",
        "fraud_pattern": "legitimate",
        "is_fraud": 0,
    }


def add_fraud_pattern(
    transaction: dict,
    profile: UserProfile,
    fake: Faker,
    rng: random.Random,
) -> dict:
    """Turn a small number of normal transactions into explainable fraud cases."""
    if rng.random() >= FRAUD_RATE:
        return transaction

    fraud_pattern = rng.choice(FRAUD_PATTERNS)
    transaction["is_fraud"] = 1
    transaction["fraud_pattern"] = fraud_pattern
    transaction["auth_method"] = "Password"

    if fraud_pattern == "high_amount":
        transaction["category"] = "tech"
        transaction["merchant_name"] = CATEGORY_RULES["tech"]["merchant"]
        transaction["amount"] = round(rng.uniform(2_000, 4_500), 2)

    elif fraud_pattern == "stolen_card":
        foreign_city = rng.choice(
            [city for city in CITIES if city != profile.home_city]
        )
        transaction["device_id"] = f"new_{fake.uuid4()[:10]}"
        transaction["merchant_city"] = foreign_city["name"]
        transaction["merchant_country"] = foreign_city["country"]
        transaction["lat"] = foreign_city["lat"]
        transaction["lon"] = foreign_city["lon"]
        transaction["amount"] = round(rng.uniform(500, 4_500), 2)

    elif fraud_pattern == "impossible_travel":
        distant_city = rng.choice(
            [city for city in CITIES if city != profile.home_city]
        )
        transaction["timestamp"] = profile.last_tx_time + timedelta(minutes=10)
        transaction["merchant_city"] = distant_city["name"]
        transaction["merchant_country"] = distant_city["country"]
        transaction["lat"] = distant_city["lat"]
        transaction["lon"] = distant_city["lon"]
        transaction["amount"] = round(rng.uniform(300, 3_000), 2)

    else:
        transaction["category"] = "entertainment"
        transaction["merchant_name"] = CATEGORY_RULES["entertainment"]["merchant"]
        transaction["channel"] = "ecommerce"
        transaction["device_id"] = f"new_{fake.uuid4()[:10]}"
        transaction["amount"] = round(rng.uniform(1, 15), 2)
        transaction["transaction_status"] = rng.choice(["approved", "declined"])

    return transaction


def generate_transactions_data(
    n_tx: int = 10_000,
    n_users: int = 500,
    seed: int = 42,
    output_dir: str | Path | None = None,
) -> pd.DataFrame:
    """Generate synthetic transactions and save them under ``data/simulated``."""
    fake = Faker()
    fake.seed_instance(seed)
    rng = random.Random(seed)

    profiles = create_user_profiles(fake, rng, n_users)
    user_ids = list(profiles)

    rows = []
    for _ in range(n_tx):
        user_id = rng.choice(user_ids)
        profile = profiles[user_id]
        transaction = generate_normal_transaction(fake, rng, profile)
        transaction = add_fraud_pattern(transaction, profile, fake, rng)
        profile.last_tx_time = transaction["timestamp"]
        rows.append(transaction)

    df = pd.DataFrame(rows)

    output_directory = ensure_directory(
        SIMULATED_DATA_DIR if output_dir is None else Path(output_dir)
    )
    output_path = output_directory / f"simulated_transactions_seed_{seed}.csv"
    df.to_csv(output_path, index=False)

    print("=" * 70)
    print(f"File path: {output_path.name}")
    print(f"Number of rows    : {len(df)}")
    print(f"Fraud rate        : {df.is_fraud.mean():.2%}")
    print("=" * 70)

    return df
