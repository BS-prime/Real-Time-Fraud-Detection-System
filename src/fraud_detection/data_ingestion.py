"""Synthetic bank/card transaction data generation.

This version is deliberately harder to model than a naive fraud simulator.
A naive simulator gives fraud a clean, disjoint feature range (e.g. fraud
amount always > 2,000; legitimate always < 800), so a model just learns a
threshold and looks unrealistically good. Real fraud detection is hard
because the fraud and legitimate distributions *overlap* -- most fraud
tries to look normal, and some normal behavior looks anomalous. Search for
"OVERLAP:" comments below for the specific places this is engineered in.
"""

from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
import csv
import logging
import random

import pandas as pd
from faker import Faker
import yaml

from fraud_detection.feature_schema import AUTH_METHODS
from fraud_detection.paths import SIMULATED_DATA_DIR, create_dir

logger = logging.getLogger(__name__)

START_DATE = datetime(2026, 1, 1, 8, 0)
DEFAULT_FRAUD_RATE = 0.015
DEFAULT_SIGNAL_NOISE_RATE = 0.15  # see _apply_signal() docstring
DEFAULT_LEGIT_ANOMALY_RATE = 0.04  # see _maybe_inject_legit_anomaly() docstring
DEFAULT_STEALTHY_FRAUD_SHARE = 0.30  # see _apply_fraud_pattern() docstring

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

# "account_takeover" is new: it deliberately mimics normal spending instead
# of forcing an unusual category/amount, so it can't be caught by amount
# thresholds alone -- see _apply_fraud_pattern().
FRAUD_PATTERNS = [
    "high_amount",
    "stolen_card",
    "impossible_travel",
    "card_testing",
    "account_takeover",
]


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
    """
    Generate realistic synthetic transactions for model training.
    """

    def __init__(
        self,
        n_users: int = 500,
        seed: int = 42,
        fraud_rate: float = DEFAULT_FRAUD_RATE,
        signal_noise_rate: float = DEFAULT_SIGNAL_NOISE_RATE,
        legit_anomaly_rate: float = DEFAULT_LEGIT_ANOMALY_RATE,
        stealthy_fraud_share: float = DEFAULT_STEALTHY_FRAUD_SHARE,
        output_dir: Path | str | None = None,
    ) -> None:
        self.n_users = n_users
        self.seed = seed
        self.fraud_rate = fraud_rate
        self.signal_noise_rate = signal_noise_rate
        self.legit_anomaly_rate = legit_anomaly_rate
        self.stealthy_fraud_share = stealthy_fraud_share
        self.output_dir = (
            Path(output_dir) if output_dir is not None else SIMULATED_DATA_DIR
        )
        self.fake = Faker()
        self.fake.seed_instance(seed)
        self.rng = random.Random(seed)
        self.profiles = self._create_user_profiles()

    def _create_user_profiles(self) -> dict[str, UserProfile]:
        """
        Build a synthetic user population with spending habits and home locations.
        """

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
        """
        Generate a valid transaction for a user profile before fraud injection.
        """

        categories = list(CATEGORY_RULES)
        category = self.rng.choices(
            categories,
            weights=SEGMENT_CATEGORY_WEIGHTS[profile.segment],
            k=1,
        )[0]
        category_rules = CATEGORY_RULES[category]

        # Capture the user's true previous transaction time *before* it
        # gets overwritten below. Fraud patterns that need to reason about
        # "time since the user's last activity" (impossible_travel,
        # account_takeover) anchor to this value rather than to
        # profile.last_tx_time, which by the time _apply_fraud_pattern
        # runs has already been advanced to *this* transaction's own
        # (pre-fraud) timestamp. Anchoring to the stale value silently
        # weakened those patterns' intended "unusually soon/far" signal.
        previous_tx_time = profile.last_tx_time
        minutes_since_last_tx = self.rng.randint(5, profile.gap_minutes * 2)
        timestamp = previous_tx_time + timedelta(minutes=minutes_since_last_tx)
        profile.last_tx_time = timestamp

        amount = round(self.rng.uniform(0.5, 1.8) * category_rules["typical_amount"], 2)
        if category in {"tech", "travel"}:
            amount = round(amount * self.rng.uniform(1.2, 2.4), 2)

        city = profile.home_city
        if category == "travel":
            city = self.rng.choice(CITIES)

        auth_method = self.rng.choice(AUTH_METHODS)
        device_id = self.rng.choice(profile.devices)

        transaction = {
            "tx_id": self.fake.uuid4()[:12],
            "timestamp": timestamp,
            "user_id": profile.user_id,
            "amount": amount,
            "category": category,
            "merchant_name": category_rules["merchant"],
            "merchant_city": city["name"],
            "merchant_country": city["country"],
            "channel": self.rng.choice(CHANNELS),
            "device_id": device_id,
            "auth_method": auth_method,
            "lat": round(city["lat"] + self.rng.uniform(-0.05, 0.05), 6),
            "lon": round(city["lon"] + self.rng.uniform(-0.05, 0.05), 6),
            "ip_address": self.fake.ipv4(),
            "customer_segment": profile.segment,
            "transaction_status": "approved",
            "fraud_pattern": "legitimate",
            "is_fraud": 0,
            # Internal-only bookkeeping field, consumed by
            # _apply_fraud_pattern and stripped in _transaction_rows
            # before a row is yielded. Never appears in output.
            "_previous_tx_time": previous_tx_time,
        }

        self._maybe_inject_legit_anomaly(transaction, profile)
        return transaction

    def _maybe_inject_legit_anomaly(
        self, transaction: dict[str, object], profile: UserProfile
    ) -> None:
        """OVERLAP: give some legitimate transactions fraud-shaped tells.

        A model trained on data where "new device" or "big purchase" only
        ever appears on fraud rows will learn those as free discriminators.
        Real customers upgrade phones, travel for work, and occasionally
        make a large one-off purchase -- all without being defrauded. This
        injects those legitimate-but-anomalous cases at `legit_anomaly_rate`
        so the model has to rely on combinations of features rather than
        any single "anomaly implies fraud" shortcut.
        """
        if self.rng.random() >= self.legit_anomaly_rate:
            return

        anomaly = self.rng.choice(["new_device", "away_from_home", "large_purchase"])

        if anomaly == "new_device":
            new_device = f"new_{self.fake.uuid4()[:10]}"
            profile.devices.append(new_device)
            transaction["device_id"] = new_device

        elif anomaly == "away_from_home":
            other_city = self.rng.choice(
                [city for city in CITIES if city != profile.home_city]
            )
            transaction["merchant_city"] = other_city["name"]
            transaction["merchant_country"] = other_city["country"]
            transaction["lat"] = round(
                other_city["lat"] + self.rng.uniform(-0.05, 0.05), 6
            )
            transaction["lon"] = round(
                other_city["lon"] + self.rng.uniform(-0.05, 0.05), 6
            )

        else:  # large_purchase
            transaction["amount"] = round(profile.avg_spend * self.rng.uniform(4, 9), 2)

    def _apply_signal(self) -> bool:
        """
        Decide whether to apply a fraud-indicative feature override.

        Without this, every fraud row got the exact same tells (forced
        auth_method, forced "new_" device prefix), which makes fraud
        trivially separable on a single feature instead of something a
        model actually has to learn. With probability `signal_noise_rate`,
        the override is skipped, so the tell is a strong signal but not a
        perfect one -- closer to how real fraud labels behave.
        """
        return self.rng.random() >= self.signal_noise_rate

    def _apply_fraud_pattern(
        self, transaction: dict[str, object], profile: UserProfile
    ) -> dict[str, object]:
        """Inject a fraud pattern into the transaction based on the configured fraud rate."""

        if self.rng.random() >= self.fraud_rate:
            return transaction

        fraud_pattern = self.rng.choice(FRAUD_PATTERNS)
        transaction["is_fraud"] = 1
        transaction["fraud_pattern"] = fraud_pattern

        # OVERLAP: a share of fraud is "stealthy" -- it keeps the amount
        # inside the customer's own normal spending range instead of
        # jumping to an obviously-fraud-sized number. Real carders often
        # deliberately keep purchases small/typical to avoid triggering
        # amount-based rules, so amount alone should not be a reliable
        # separator.
        is_stealthy = self.rng.random() < self.stealthy_fraud_share

        if self._apply_signal():
            transaction["auth_method"] = "Password"

        if fraud_pattern == "high_amount":
            transaction["category"] = "tech"
            transaction["merchant_name"] = CATEGORY_RULES["tech"]["merchant"]
            if is_stealthy:
                transaction["amount"] = round(
                    CATEGORY_RULES["tech"]["typical_amount"]
                    * self.rng.uniform(0.8, 2.0),
                    2,
                )
            else:
                transaction["amount"] = round(self.rng.uniform(2_000, 4_500), 2)

        elif fraud_pattern == "stolen_card":
            foreign_city = self.rng.choice(
                [city for city in CITIES if city != profile.home_city]
            )
            if self._apply_signal():
                transaction["device_id"] = f"new_{self.fake.uuid4()[:10]}"
            transaction["merchant_city"] = foreign_city["name"]
            transaction["merchant_country"] = foreign_city["country"]
            transaction["lat"] = foreign_city["lat"]
            transaction["lon"] = foreign_city["lon"]
            transaction["amount"] = round(
                profile.avg_spend * self.rng.uniform(1.0, 3.0)
                if is_stealthy
                else self.rng.uniform(500, 4_500),
                2,
            )

        elif fraud_pattern == "impossible_travel":
            distant_city = self.rng.choice(
                [city for city in CITIES if city != profile.home_city]
            )
            # OVERLAP: a fixed 10-minute gap makes "impossible travel" a
            # trivial time-delta rule. Real geo-velocity fraud sometimes
            # leaves a gap wide enough to *look* physically plausible
            # (a few hours), so catching it requires reasoning about
            # distance vs. elapsed time rather than a single threshold.
            #
            # Anchored to the user's true previous transaction time
            # (transaction["_previous_tx_time"]), not profile.last_tx_time,
            # which at this point already reflects the current (pre-fraud)
            # transaction rather than the prior one.
            gap_minutes = (
                self.rng.randint(6, 45)
                if not is_stealthy
                else self.rng.randint(120, 360)
            )
            transaction["timestamp"] = transaction["_previous_tx_time"] + timedelta(
                minutes=gap_minutes
            )
            transaction["merchant_city"] = distant_city["name"]
            transaction["merchant_country"] = distant_city["country"]
            transaction["lat"] = distant_city["lat"]
            transaction["lon"] = distant_city["lon"]
            transaction["amount"] = round(self.rng.uniform(300, 3_000), 2)

        elif fraud_pattern == "card_testing":
            transaction["category"] = "entertainment"
            transaction["merchant_name"] = CATEGORY_RULES["entertainment"]["merchant"]
            transaction["channel"] = "ecommerce"
            if self._apply_signal():
                transaction["device_id"] = f"new_{self.fake.uuid4()[:10]}"
            transaction["amount"] = round(self.rng.uniform(1, 15), 2)
            transaction["transaction_status"] = self.rng.choice(
                ["approved", "declined"]
            )

        else:  # account_takeover
            # OVERLAP: deliberately does NOT override category, amount,
            # city, or channel. This is a fraudster operating a
            # compromised account carefully -- the only differences from
            # genuine behavior are a device change (itself noised by
            # _apply_signal) and a shortened gap since the last
            # transaction, both weak signals rather than hard rules.
            #
            # Anchored to transaction["_previous_tx_time"] for the same
            # reason as impossible_travel above -- otherwise the
            # "shortened gap" signal is computed relative to a timestamp
            # that was itself only just generated for this row, which
            # dilutes the intended "unusually soon after last activity"
            # tell.
            if self._apply_signal():
                transaction["device_id"] = f"new_{self.fake.uuid4()[:10]}"
            shortened_gap = max(
                5, int(profile.gap_minutes * self.rng.uniform(0.1, 0.4))
            )
            transaction["timestamp"] = transaction["_previous_tx_time"] + timedelta(
                minutes=shortened_gap
            )

        return transaction

    def _output_path(self) -> Path:
        directory = create_dir(self.output_dir)
        return directory / f"simulated_transactions_seed_{self.seed}.csv"

    def _transaction_columns(self) -> list[str]:
        """
        Define the column order for the output CSV. This ensures consistent ordering regardless of how dictionaries are iterated or how DataFrames are constructed.
        """

        return [
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
        ]

    def _transaction_rows(self, n_tx: int = 1_000_000) :
        """
        Generate transaction rows as dictionaries, yielding one at a time to avoid
        holding the entire dataset in memory. This is useful for generating very large datasets without running out of memory.
        """

        for _ in range(n_tx):
            profile = self.profiles[self.rng.choice(list(self.profiles))]
            transaction = self._generate_normal_transaction(profile)
            transaction = self._apply_fraud_pattern(transaction, profile)
            profile.last_tx_time = transaction["timestamp"]
            # Drop internal bookkeeping field before this row is exposed
            # to callers/CSV output.
            del transaction["_previous_tx_time"]
            yield transaction

    def generate(self, n_tx: int = 10_000) -> pd.DataFrame:
        """
        Generate a DataFrame of synthetic transactions.
        """
        if n_tx <= 0:
            raise ValueError("n_tx must be a positive integer")
        
        if n_tx > 100_000:
            logger.warning(
                "Generating a very large number of transactions (%d). "
                "This may take a while and consume significant memory.",
                n_tx,
            )
        return pd.DataFrame(self._transaction_rows(n_tx))

    def save(
        self, transactions: pd.DataFrame | list[dict[str, object]] | object
    ) -> Path:
        """
        Save the generated transactions to a CSV file.
        """

        output_path = self._output_path()

        # If the transactions are already a DataFrame, use its built-in CSV writer.
        if isinstance(transactions, pd.DataFrame):
            transactions.to_csv(output_path, index=False)
            row_count = len(transactions)

        # If the transactions are a list of dictionaries, write them manually with csv.DictWriter.
        else:
            rows_written = 0
            fieldnames = self._transaction_columns()
            with output_path.open("w", newline="", encoding="utf-8") as stream:
                writer = csv.DictWriter(stream, fieldnames=fieldnames)
                writer.writeheader()
                for row in transactions:
                    writer.writerow({key: row[key] for key in fieldnames})
                    rows_written += 1
            row_count = rows_written

        logger.info(
            "Saved simulated transaction data to %s with %d rows",
            output_path,
            row_count,
        )
        return output_path

    def generate_transactions_data(
        self,
        n_tx: int = 1_000_000,
    ) -> Path:
        """
        Generate simulated transaction data and save it to a CSV file.
        """

        return self.save(self._transaction_rows(n_tx))


def generate_transactions_data(
    n_tx: int = 1_000_000,
    n_users: int = 5_000,
    seed: int = 42,
    output_dir: str | Path | None = None,
) -> Path:
    """
    Generate synthetic transactions and save them under `data/simulated` directory.
    """

    simulator = TransactionSimulator(
        n_users=n_users,
        seed=seed,
        output_dir=output_dir,
    )
    return simulator.generate_transactions_data(n_tx=n_tx)

def load_params(path: str | Path = Path("configs/params.yaml")) -> dict[str, object]:
    """
    Load parameters from a YAML file.
    """

    ROOT_DIR = Path(__file__).parents[2]
    path = ROOT_DIR / path

    with open(path, "r", encoding="utf-8") as f:
        params = yaml.safe_load(f)

    return params
if __name__ == "__main__":

    params = load_params()["data_ingestion"]
    
    generate_transactions_data(
        n_tx=params["n_tx"], 
        n_users=params["n_users"], 
        seed=params["seed"],
    )
    
